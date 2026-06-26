"""
Chunk embedder and database storer for IRS document ingestion.

Calls Ollama's /api/embed endpoint in batches, then persists chunks and
their embeddings to PostgreSQL using batch inserts.
"""

import logging
import os
import re
import time
from dataclasses import dataclass

import httpx
import psycopg2
import psycopg2.extensions
from psycopg2.extras import execute_values
from dotenv import load_dotenv

from ingestion.chunker import Chunk

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL: str = os.environ["OLLAMA_BASE_URL"]
EMBED_MODEL: str = os.environ["EMBED_MODEL"]

EMBED_ENDPOINT: str = "/api/embed"
DEFAULT_BATCH_SIZE: int = 32
EMBED_TIMEOUT: float = 120.0
MAX_RETRIES: int = 3
RETRY_DELAYS: list[int] = [1, 2, 4]

TOKEN_SPLIT_RE: re.Pattern = re.compile(r"\W+")

INSERT_CHUNKS_SQL = """
INSERT INTO chunks
    (content, embedding, source_doc, article, section,
     page_number, fiscal_year, chunk_index)
VALUES %s
RETURNING id
"""

INSERT_BM25_SQL = "INSERT INTO bm25_corpus (chunk_id, tokens) VALUES %s"


@dataclass
class ChunkWithEmbedding:
    """A Chunk paired with its embedding vector.

    Args:
        content: Text content of the chunk.
        source_doc: Source document name.
        article: Article identifier, or None.
        section: Section/chapter heading, or None.
        page_number: Starting page in the source PDF, or None.
        fiscal_year: Fiscal year this chunk belongs to.
        chunk_index: Sequential index within the document.
        embedding: 768-dimensional float vector from nomic-embed-text.
    """

    content: str
    source_doc: str
    article: str | None
    section: str | None
    page_number: int | None
    fiscal_year: int
    chunk_index: int
    embedding: list[float]


def embed_chunks(
    chunks: list[Chunk],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[ChunkWithEmbedding]:
    """Embed a list of Chunks by calling Ollama's /api/embed endpoint.

    Processes chunks in batches of batch_size. Retries up to MAX_RETRIES times
    on HTTP or connection errors with exponential backoff.

    Args:
        chunks: List of Chunk objects whose content will be embedded.
        batch_size: Number of chunks to send per Ollama request.

    Returns:
        List of ChunkWithEmbedding objects in the same order as the input.

    Raises:
        httpx.HTTPError: if a batch fails after all retries.
        httpx.RequestError: if Ollama is unreachable after all retries.
        KeyError: if the Ollama response is missing the 'embeddings' key.
    """
    results: list[ChunkWithEmbedding] = []
    total = len(chunks)
    url = f"{OLLAMA_BASE_URL}{EMBED_ENDPOINT}"

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c.content for c in batch]
        payload = {"model": EMBED_MODEL, "input": texts}

        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = httpx.post(url, json=payload, timeout=EMBED_TIMEOUT)
                response.raise_for_status()
                last_exc = None
                break
            except (httpx.HTTPError, httpx.RequestError) as exc:
                last_exc = exc
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    "Embed attempt %d/%d failed: %s. Retrying in %ds…",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)

        if last_exc is not None:
            raise last_exc

        embeddings: list[list[float]] = response.json()["embeddings"]
        for chunk, embedding in zip(batch, embeddings):
            results.append(
                ChunkWithEmbedding(
                    content=chunk.content,
                    source_doc=chunk.source_doc,
                    article=chunk.article,
                    section=chunk.section,
                    page_number=chunk.page_number,
                    fiscal_year=chunk.fiscal_year,
                    chunk_index=chunk.chunk_index,
                    embedding=embedding,
                )
            )

        embedded_so_far = min(batch_start + batch_size, total)
        print(f"Embedded {embedded_so_far}/{total} chunks")

    return results


def store_chunks(
    chunks_with_embeddings: list[ChunkWithEmbedding],
    conn: psycopg2.extensions.connection,
) -> int:
    """Batch-insert chunks and their BM25 tokens into PostgreSQL.

    Inserts into chunks first (getting back generated IDs), then inserts the
    tokenised content into bm25_corpus using those IDs. The FK order is
    mandatory — bm25_corpus.chunk_id references chunks.id.

    execute_values with fetch=True returns RETURNING rows in input order,
    making it safe to zip chunk_ids with the original chunks list.

    Args:
        chunks_with_embeddings: List of ChunkWithEmbedding objects to store.
        conn: Open psycopg2 connection. The caller owns this connection and
              must close it; this function commits via the context manager.

    Returns:
        Number of rows inserted into the chunks table.

    Raises:
        psycopg2.Error: on any database error (transaction is rolled back).
    """
    with conn:
        with conn.cursor() as cur:
            chunk_rows = [
                (
                    c.content,
                    "[" + ",".join(str(v) for v in c.embedding) + "]",
                    c.source_doc,
                    c.article,
                    c.section,
                    c.page_number,
                    c.fiscal_year,
                    c.chunk_index,
                )
                for c in chunks_with_embeddings
            ]

            inserted = execute_values(
                cur,
                INSERT_CHUNKS_SQL,
                chunk_rows,
                template="(%s, %s::vector, %s, %s, %s, %s, %s, %s)",
                fetch=True,
            )
            chunk_ids = [row[0] for row in inserted]

            bm25_rows = [
                (
                    chunk_id,
                    " ".join(t for t in TOKEN_SPLIT_RE.split(c.content.lower()) if t),
                )
                for chunk_id, c in zip(chunk_ids, chunks_with_embeddings)
            ]

            execute_values(cur, INSERT_BM25_SQL, bm25_rows)

    return len(chunk_ids)
