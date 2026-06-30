"""
Interactive REPL for the IRS Assistant.

Wires together retrieval and generation into a terminal chat loop.
The BM25 index and DB connection are initialised once at startup and
reused for every question. Tokens stream to stdout as they arrive.
"""

import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL: str = os.environ["POSTGRES_URL"]
TOP_K: int = int(os.environ.get("TOP_K", 5))
VECTOR_WEIGHT: float = float(os.environ.get("VECTOR_WEIGHT", 0.7))
BM25_WEIGHT: float = float(os.environ.get("BM25_WEIGHT", 0.3))

from generation.client_factory import get_generation_client
from generation.errors import LLMConnectionError, LLMRateLimitError
from generation.prompt import build_prompt
from retrieval.bm25_search import build_bm25_index, bm25_search
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.vector_search import embed_query, vector_search

BANNER = """
Assistente IRS (Portugal)
Documentos: Código do IRS 2024, Guia de Deduções IRS 2024
Escreve 'sair', 'exit' ou 'quit' para terminar.
"""
EXIT_WORDS = {"sair", "exit", "quit"}


def main() -> None:
    client = get_generation_client()

    if not client.health_check():
        print("Erro: não foi possível ligar ao modelo de linguagem. Verifica a tua configuração.")
        sys.exit(1)

    try:
        conn = psycopg2.connect(POSTGRES_URL)
    except psycopg2.OperationalError as exc:
        print(f"Erro ao ligar à base de dados: {exc}")
        sys.exit(1)

    try:
        bm25_index, chunk_ids = build_bm25_index(conn)
        print(BANNER)

        while True:
            try:
                question = input("Pergunta: ").strip()
            except KeyboardInterrupt:
                print("\nAté logo.")
                break

            if not question:
                continue
            if question.lower() in EXIT_WORDS:
                print("Até logo.")
                break

            query_embedding = embed_query(question)
            vector_results = vector_search(query_embedding, conn, top_k=TOP_K * 2)
            bm25_results = bm25_search(question, bm25_index, chunk_ids, conn, top_k=TOP_K * 2)
            chunks = reciprocal_rank_fusion(
                vector_results, bm25_results, VECTOR_WEIGHT, BM25_WEIGHT
            )[:TOP_K]

            prompt = build_prompt(question, chunks)
            print("\nResposta: ", end="", flush=True)
            try:
                for token in client.generate(prompt, stream=True):
                    print(token, end="", flush=True)
            except LLMRateLimitError as exc:
                wait = int(exc.retry_after) if exc.retry_after else None
                wait_msg = f" Aguarda {wait}s ou" if wait else ""
                print(f"\nLimite de pedidos Groq atingido.{wait_msg} corre o modelo localmente para não teres limite.")
            except LLMConnectionError as exc:
                print(f"\nErro ao ligar ao modelo: {exc}")
            print("\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
