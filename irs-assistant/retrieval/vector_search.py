"""
Vector similarity search over the chunks table using pgvector.

Provides query embedding via Ollama and cosine similarity search using
the HNSW index on the embedding column.
"""

import logging
import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

import psycopg2.extensions

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL: str = os.environ["OLLAMA_BASE_URL"]
EMBED_MODEL: str = os.environ["EMBED_MODEL"]
EMBED_ENDPOINT: str = "/api/embed"
EMBED_TIMEOUT: float = 30.0
MAX_RETRIES: int = 3
RETRY_DELAYS: list[int] = [1, 2, 4]
OLLAMA_ERROR_MESSAGE: str = "Ollama não está a correr. Inicia com: ollama serve"

VECTOR_SEARCH_SQL = """
SELECT id, content, source_doc, article, page_number, fiscal_year,
       1 - (embedding <=> %s::vector) AS score
FROM chunks
ORDER BY embedding <=> %s::vector
LIMIT %s
"""

VECTOR_SEARCH_FILTERED_SQL = """
SELECT id, content, source_doc, article, page_number, fiscal_year,
       1 - (embedding <=> %s::vector) AS score
FROM chunks
WHERE fiscal_year = %s
ORDER BY embedding <=> %s::vector
LIMIT %s
"""


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama is unreachable after all retry attempts."""


@dataclass
class SearchResult:
    """A single retrieved chunk with its relevance score.

    Args:
        chunk_id: Primary key of the chunks table row.
        content: Text content of the chunk.
        source_doc: Human-readable source document name.
        article: Article identifier (e.g. "Artigo 12.º"), or None.
        page_number: 1-indexed starting page in the source PDF, or None.
        fiscal_year: Fiscal year the chunk belongs to.
        score: Relevance score. For vector search: cosine similarity in [0,1].
               For BM25: normalised BM25 score in [0,1].
               For RRF fusion: accumulated RRF score (small positive float).
    """

    chunk_id: int
    content: str
    source_doc: str
    article: str | None
    page_number: int | None
    fiscal_year: int
    score: float


def embed_query(text: str) -> list[float]:
    """Embed a single query string using Ollama's nomic-embed-text model.

    Args:
        text: The query string to embed.

    Returns:
        768-dimensional float vector representing the query.

    Raises:
        OllamaConnectionError: if Ollama is unreachable after MAX_RETRIES attempts.
    """
    url = f"{OLLAMA_BASE_URL}{EMBED_ENDPOINT}"
    payload = {"model": EMBED_MODEL, "input": [text]}

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = httpx.post(url, json=payload, timeout=EMBED_TIMEOUT)
            response.raise_for_status()
            return response.json()["embeddings"][0]
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            last_exc = exc
            logger.warning(
                "embed_query attempt %d/%d failed: %s. Retrying in %ds…",
                attempt + 1,
                MAX_RETRIES,
                exc,
                RETRY_DELAYS[attempt],
            )
            time.sleep(RETRY_DELAYS[attempt])

    raise OllamaConnectionError(OLLAMA_ERROR_MESSAGE) from last_exc


def vector_search(
    query_embedding: list[float],
    conn: psycopg2.extensions.connection,
    top_k: int = 10,
    fiscal_year: int | None = None,
) -> list[SearchResult]:
    """Search the chunks table by cosine similarity to a query embedding.

    Uses pgvector's <=> cosine distance operator and the HNSW index.
    Score = 1 - cosine_distance, so higher scores mean more similar.

    Args:
        query_embedding: 768-dimensional float vector from embed_query.
        conn: Open psycopg2 connection. Caller owns lifecycle; this function
              does not commit or close the connection.
        top_k: Maximum number of results to return.
        fiscal_year: If provided, restricts results to this fiscal year.

    Returns:
        List of SearchResult sorted by score descending (most similar first).

    Raises:
        psycopg2.Error: on database errors.
    """
    emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    if fiscal_year is None:
        sql = VECTOR_SEARCH_SQL
        params = (emb_str, emb_str, top_k)
    else:
        sql = VECTOR_SEARCH_FILTERED_SQL
        params = (emb_str, fiscal_year, emb_str, top_k)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        SearchResult(
            chunk_id=row[0],
            content=row[1],
            source_doc=row[2],
            article=row[3],
            page_number=row[4],
            fiscal_year=row[5],
            score=float(row[6]),
        )
        for row in rows
    ]
