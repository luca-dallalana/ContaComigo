"""
Streamlit web UI for the IRS Assistant.

Resources (DB connection, BM25 index, generation client) are initialised once
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
INFERENCE_BACKEND: str = os.environ.get("INFERENCE_BACKEND", "ollama").lower()
GENERATION_MODEL: str = os.environ.get("GENERATION_MODEL", "llama3.1:8b")

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

EXAMPLE_QUESTIONS: list[str] = [
    "Até quando posso entregar o IRS em 2026?",
    "Quais são as despesas de saúde dedutíveis?",
    "Como funciona o regime simplificado?",
    "O que é o mínimo de existência?",
]

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&family=DM+Serif+Display&display=swap');

h1, h2, h3 {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-weight: 400 !important;
    letter-spacing: -0.01em !important;
}

.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessageContent"] p,
section[data-testid="stSidebar"],
[data-testid="stChatInput"] textarea,
.stButton > button, label, .stCaption {
    font-family: 'DM Sans', system-ui, sans-serif !important;
}

.stButton > button {
    border-radius: 2px !important;
}

[data-testid="stChatInput"] textarea {
    border-radius: 2px !important;
}

[data-testid="stChatMessageAvatarCustom"],
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
}

[data-testid="stChatMessage"] { background-color: transparent !important; }
[data-testid="stChatMessageContent"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatMessageAvatarAssistant"] ~ [data-testid="stChatMessageContent"] {
    border-left: 2px solid #B5800E !important;
    padding-left: 18px !important;
}

[data-testid="stChatInput"] {
    box-shadow: 0 0 15px rgba(0,0,0,0.06) !important;
    transition: box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    box-shadow: 0 0 25px rgba(0,0,0,0.13) !important;
}
</style>
"""

DARK_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"] { background-color: #1A1714 !important; }
header[data-testid="stHeader"], [data-testid="stHeader"] { background-color: #1A1714 !important; }
section[data-testid="stSidebar"] > div:first-child { background-color: #201B16 !important; }
[data-testid="stChatMessage"] { background-color: transparent !important; }
[data-testid="stChatMessageAvatarCustom"],
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
}
[data-testid="stChatMessageContent"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatMessageAvatarAssistant"] ~ [data-testid="stChatMessageContent"] {
    border-left: 2px solid #B5800E !important;
    padding-left: 18px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #251F1A !important;
    color: #EDE5D8 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #7A6E65 !important; }
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    background-color: #251F1A !important;
    border-color: #3D3228 !important;
    border-width: 1px !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] button {
    background-color: #B5800E !important;
    color: #FFFFFF !important;
    border-color: #B5800E !important;
}
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > div,
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > div,
[data-testid="stChatInputContainer"],
[data-testid="stChatInputContainer"] > div {
    background-color: #1A1714 !important;
    border-color: #3D3228 !important;
    box-shadow: none !important;
}
.stButton > button { background-color: #251F1A !important; color: #C5B8A8 !important; border-color: #3D3228 !important; }
[data-testid="stExpander"] { background-color: #251F1A !important; border-color: #3D3228 !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > summary,
[data-testid="stExpander"] details > summary,
details[data-testid="stExpander"] > summary {
    background-color: #251F1A !important;
    color: #C5B8A8 !important;
    border-color: #3D3228 !important;
}
[data-testid="stExpanderDetails"] { background-color: #251F1A !important; }
hr { border-color: #3D3228 !important; }
p, li, label, .stCaption, .stMarkdown { color: #C5B8A8 !important; }
h1, h2, h3 { color: #F5EDE0 !important; }
code { background-color: #251F1A !important; color: #D4A840 !important; }
a { color: #D4A840 !important; }
section[data-testid="stSidebar"] > div > div:first-child,
section[data-testid="stSidebar"] > div > div:first-child *,
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] * {
    opacity: 1 !important;
}
section[data-testid="stSidebar"] > div > div:first-child svg,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}
</style>
"""


def _pdf_link(source_doc: str, page_number: int | None) -> str | None:
    base = SOURCE_PDF_URLS.get(source_doc)
    if not base:
        return None
    return f"{base}#page={page_number}" if page_number else base


def _render_source_card(idx: int, chunk, dark: bool = False) -> None:
    label = chunk.article or "Secção geral"
    pdf_link = _pdf_link(chunk.source_doc, chunk.page_number)
    page_str = f" &nbsp;·&nbsp; p.&nbsp;{chunk.page_number}" if chunk.page_number else ""
    content_escaped = chunk.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    card_bg = "#251F1A" if dark else "#F5EDE0"
    content_color = "#A89880" if dark else "#4A4239"
    link_color = "#D4A840" if dark else "#8B6210"
    mono = "'DM Mono','Courier New',monospace"
    # neubrutalism offset shadow from 21st.dev card variants
    shadow = "3px 3px 0px 0px rgba(181,128,14,0.28)" if not dark else "3px 3px 0px 0px rgba(181,128,14,0.18)"
    border = "0.5px solid rgba(181,128,14,0.45)" if not dark else "0.5px solid rgba(181,128,14,0.3)"

    if pdf_link:
        title_html = (
            f'<a href="{pdf_link}" target="_blank" '
            f'style="color:{link_color};text-decoration:none;font-family:{mono};font-size:0.78em;letter-spacing:0.03em;">'
            f'[{idx}] {chunk.source_doc} — {label}</a>{page_str}'
        )
    else:
        title_html = (
            f'<span style="font-family:{mono};font-size:0.78em;letter-spacing:0.03em;color:{link_color};">'
            f'[{idx}] {chunk.source_doc} — {label}</span>{page_str}'
        )

    st.markdown(
        f"""
        <div style="border:{border};padding:10px 14px;margin:8px 0;
                    background-color:{card_bg};box-shadow:{shadow};">
            <div style="margin-bottom:6px;">{title_html}</div>
            <div style="font-size:0.82em;color:{content_color};line-height:1.6;">{content_escaped}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
st.markdown(BASE_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Assistente IRS")
    st.caption("Informação fiscal portuguesa com base em documentos oficiais da AT.")
    st.divider()
    dark_mode = st.toggle("Modo escuro", value=True)
    st.divider()
    backend_label = "Groq Cloud" if INFERENCE_BACKEND == "groq" else "Ollama (local)"
    st.markdown(f"**Modelo:** `{GENERATION_MODEL}`")
    st.markdown(f"**Backend:** {backend_label}")
    st.divider()
    st.markdown("**Documentos:**")
    st.markdown(
        "- Código do IRS (CIRS) 2025\n"
        "- Estatuto dos Benefícios Fiscais 2025\n"
        "- Guia de Deduções IRS 2025\n"
        "- Principais Prazos IRS 2025\n"
        "- Ofício Circulado 20269/2024"
    )
    st.divider()
    st.caption(
        "Esta informação é de carácter geral e não substitui aconselhamento fiscal profissional. "
        "Para a tua situação específica, consulta um contabilista certificado ou o "
        "[apoio ao contribuinte da AT](https://www.portaldasfinancas.gov.pt)."
    )

if dark_mode:
    st.markdown(DARK_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <h1 style="font-family:'DM Serif Display',Georgia,serif;font-weight:400;
               letter-spacing:-0.01em;margin-bottom:0.25rem;">
      <span style="position:relative;display:inline-block;padding-bottom:10px;">
        Assistente IRS
        <svg style="position:absolute;width:110%;height:18px;bottom:-1px;left:-5%;"
             viewBox="0 0 140 22" preserveAspectRatio="none">
          <path d="M4 15 Q 70 22, 136 13" stroke="#B5800E" stroke-width="2.5"
                stroke-linecap="round" fill="none"/>
        </svg>
      </span>
    </h1>
    """,
    unsafe_allow_html=True,
)
st.caption("Código do IRS 2025 · Guia de Deduções IRS 2025")

client, conn, bm25_index, chunk_ids = _get_resources()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rated" not in st.session_state:
    st.session_state["rated"] = set()

if not st.session_state["messages"] and "pending_question" not in st.session_state:
    st.markdown(
        "Olá! Sou um assistente especializado em IRS português. "
        "Posso ajudar-te com questões sobre deduções, categorias de rendimento, prazos e muito mais."
    )
    st.markdown("**Experimenta uma destas perguntas:**")
    col1, col2 = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with (col1 if i % 2 == 0 else col2):
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state["pending_question"] = q
                st.rerun()

for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if msg["role"] == "assistant" and i not in st.session_state["rated"]:
        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"up_{i}", help="Resposta útil"):
                save_feedback(FEEDBACK_DB, msg["question"], msg["content"], msg["sources"], 1)
                st.session_state["rated"].add(i)
                st.rerun()
        with col2:
            if st.button("👎", key=f"down_{i}", help="Resposta não útil"):
                save_feedback(FEEDBACK_DB, msg["question"], msg["content"], msg["sources"], -1)
                st.session_state["rated"].add(i)
                st.rerun()

question = st.chat_input("Faz uma pergunta sobre IRS...")
if not question:
    question = st.session_state.pop("pending_question", None)

if question:
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
            st.error("Não foi possível gerar a pesquisa. Verifica se o Ollama está a correr: `ollama serve`")
        st.stop()

    prompt = build_prompt(question, chunks)

    with st.chat_message("assistant"):
        try:
            response_text = st.write_stream(client.generate(prompt, stream=True))
        except LLMRateLimitError as exc:
            wait = int(exc.retry_after) if exc.retry_after else None
            wait_msg = f" Podes aguardar {wait} segundos ou" if wait else ""
            st.error(
                f"Limite de pedidos da API Groq atingido.{wait_msg} corre o modelo localmente para não teres esperas (`INFERENCE_BACKEND=ollama`)."
            )
            response_text = ""
        except LLMConnectionError as exc:
            st.error(f"Não foi possível obter resposta do modelo de IA. Tenta novamente em instantes. ({exc})")
            response_text = ""

    if chunks:
        with st.expander("Fontes consultadas"):
            for idx, chunk in enumerate(chunks, start=1):
                _render_source_card(idx, chunk, dark=dark_mode)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": response_text,
        "question": question,
        "sources": [
            f"{c.source_doc} — {c.article or 'Secção geral'}" for c in chunks
        ],
    })
