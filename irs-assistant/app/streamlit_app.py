"""
Streamlit web UI for the IRS Assistant.

Resources (DB connection, BM25 index, OllamaClient) are initialised once
per server process via st.cache_resource and reused across all interactions.
Chat history is stored in st.session_state and replayed on each script re-run.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL: str = os.environ["POSTGRES_URL"]
TOP_K: int = int(os.environ.get("TOP_K", 5))
VECTOR_WEIGHT: float = float(os.environ.get("VECTOR_WEIGHT", 0.7))
BM25_WEIGHT: float = float(os.environ.get("BM25_WEIGHT", 0.3))

from feedback.store import init_db, save_feedback
from generation.client_factory import get_generation_client
from generation.errors import LLMConnectionError, LLMRateLimitError
from generation.prompt import build_prompt
from retrieval.bm25_search import bm25_search, build_bm25_index
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.vector_search import LLMConnectionError as EmbedConnectionError
from retrieval.vector_search import embed_query, vector_search

FEEDBACK_DB: str = os.environ.get("FEEDBACK_DB", "feedback.sqlite")
init_db(FEEDBACK_DB)

SOURCE_PDF_URLS: dict[str, str] = {
    "Código do IRS (CIRS)": "https://info.portaldasfinancas.gov.pt/pt/informacao_fiscal/codigos_tributarios/Cod_download/Documents/CIRS.pdf",
    "Estatuto dos Benefícios Fiscais (EBF)": "https://info.portaldasfinancas.gov.pt/pt/informacao_fiscal/codigos_tributarios/Cod_download/Documents/EBF.pdf",
    "Guia de Deduções IRS 2025": "https://info.portaldasfinancas.gov.pt/pt/apoio_contribuinte/Folhetos_informativos/Documents/IRS_deducoes_2025.pdf",
    "Principais Prazos IRS 2025": "https://info.portaldasfinancas.gov.pt/pt/apoio_contribuinte/Folhetos_informativos/Documents/IRS_2025_Principais_prazos_2026.pdf",
    "Ofício Circulado 20269/2024 (IRS)": "https://info.portaldasfinancas.gov.pt/pt/informacao_fiscal/legislacao/instrucoes_administrativas/Documents/Oficio_circulado_20269_2024.pdf",
}


def _pdf_link(source_doc: str, page_number: int | None) -> str | None:
    base = SOURCE_PDF_URLS.get(source_doc)
    if not base:
        return None
    return f"{base}#page={page_number}" if page_number else base


@st.cache_resource
def _get_resources():
    client = get_generation_client()
    if not client.health_check():
        st.error("O modelo de IA não está acessível. Verifica as tuas credenciais ou se o serviço está a correr.")
        st.stop()
    try:
        conn = psycopg2.connect(POSTGRES_URL)
    except psycopg2.OperationalError as exc:
        st.error(f"Não foi possível ligar à base de dados. Verifica se o PostgreSQL está a correr. ({exc})")
        st.stop()
    bm25_index, chunk_ids = build_bm25_index(conn)
    return client, conn, bm25_index, chunk_ids


st.set_page_config(page_title="Assistente IRS", layout="centered")
st.title("Assistente IRS (Portugal)")
st.caption("Código do IRS 2025 · Guia de Deduções IRS 2025")

client, conn, bm25_index, chunk_ids = _get_resources()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rated" not in st.session_state:
    st.session_state["rated"] = set()

for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if msg["role"] == "assistant" and i not in st.session_state["rated"]:
        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("Util", key=f"up_{i}"):
                save_feedback(FEEDBACK_DB, msg["question"], msg["content"], msg["sources"], 1)
                st.session_state["rated"].add(i)
                st.rerun()
        with col2:
            if st.button("Nao util", key=f"down_{i}"):
                save_feedback(FEEDBACK_DB, msg["question"], msg["content"], msg["sources"], -1)
                st.session_state["rated"].add(i)
                st.rerun()

if question := st.chat_input("Faz uma pergunta sobre IRS..."):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        query_embedding = embed_query(question)
        vector_results = vector_search(query_embedding, conn, top_k=TOP_K * 2)
        bm25_results = bm25_search(question, bm25_index, chunk_ids, conn, top_k=TOP_K * 2)
        chunks = reciprocal_rank_fusion(
            vector_results, bm25_results, VECTOR_WEIGHT, BM25_WEIGHT
        )[:TOP_K]
    except EmbedConnectionError:
        with st.chat_message("assistant"):
            st.error("Não foi possível gerar a pesquisa. Verifica se o Ollama está a correr (ollama serve).")
        st.stop()

    prompt = build_prompt(question, chunks)

    with st.chat_message("assistant"):
        try:
            response_text = st.write_stream(client.generate(prompt, stream=True))
        except LLMRateLimitError as exc:
            wait = int(exc.retry_after) if exc.retry_after else None
            wait_msg = f" Podes aguardar {wait} segundos" if wait else ""
            st.error(
                f"Limite de pedidos da API Groq atingido.{wait_msg} ou corre o modelo localmente para não teres esperas (INFERENCE_BACKEND=ollama)."
            )
            response_text = ""
        except LLMConnectionError as exc:
            st.error(f"Não foi possível obter resposta do modelo de IA. Tenta novamente em instantes. ({exc})")
            response_text = ""

    if chunks:
        with st.expander("Fontes consultadas"):
            for idx, chunk in enumerate(chunks, start=1):
                label = chunk.article or "Secção geral"
                pdf_link = _pdf_link(chunk.source_doc, chunk.page_number)
                if pdf_link:
                    header = f"**[{idx}] [{chunk.source_doc} — {label}]({pdf_link})**"
                    if chunk.page_number:
                        header += f" *(p. {chunk.page_number})*"
                else:
                    header = f"**[{idx}] {chunk.source_doc} — {label}**"
                st.markdown(header)
                st.caption(chunk.content)
                if idx < len(chunks):
                    st.divider()

    st.session_state["messages"].append({
        "role": "assistant",
        "content": response_text,
        "question": question,
        "sources": [
            f"{c.source_doc} — {c.article or 'Secção geral'}" for c in chunks
        ],
    })
