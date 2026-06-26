"""
Database initialisation script.

Creates the pgvector extension, the chunks table with HNSW index,
and the bm25_corpus cache table.

Usage:
    python scripts/init_db.py

Raises:
    psycopg2.OperationalError: if the database is unreachable after all retries.
    psycopg2.Error: on any other database error.
"""

import logging
import os
import time
from typing import Optional

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

POSTGRES_URL: str = os.environ["POSTGRES_URL"]

MAX_RETRIES: int = 10
RETRY_INTERVAL_SECONDS: float = 2.0

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768),
    source_doc TEXT NOT NULL,
    article TEXT,
    section TEXT,
    page_number INTEGER,
    fiscal_year INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

CREATE_HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

CREATE_FISCAL_YEAR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_fiscal_year_idx
    ON chunks (fiscal_year);
"""

CREATE_SOURCE_DOC_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_source_doc_idx
    ON chunks (source_doc);
"""

CREATE_BM25_CORPUS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bm25_corpus (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES chunks(id),
    tokens TEXT NOT NULL
);
"""


def _connect_with_retry(url: str) -> psycopg2.extensions.connection:
    """
    Attempt to connect to PostgreSQL, retrying on failure.

    Docker containers take a few seconds to accept connections after
    `docker compose up`, so this retry loop prevents spurious startup failures.

    Args:
        url: libpq-compatible connection string.

    Returns:
        An open psycopg2 connection.

    Raises:
        psycopg2.OperationalError: if the database is still unreachable after
            MAX_RETRIES attempts.
    """
    last_error: Optional[psycopg2.OperationalError] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = psycopg2.connect(url)
            logger.info("Connected to database on attempt %d.", attempt)
            return conn
        except psycopg2.OperationalError as exc:
            last_error = exc
            logger.warning(
                "Connection attempt %d/%d failed: %s. Retrying in %.0fs…",
                attempt,
                MAX_RETRIES,
                exc,
                RETRY_INTERVAL_SECONDS,
            )
            time.sleep(RETRY_INTERVAL_SECONDS)
    raise psycopg2.OperationalError(
        f"Could not connect to database after {MAX_RETRIES} attempts."
    ) from last_error


def init_db() -> None:
    """
    Initialise the database schema.

    Enables the pgvector extension, creates the chunks table with its HNSW and
    B-tree indexes, and creates the bm25_corpus cache table. All statements are
    idempotent (IF NOT EXISTS), so re-running is safe.

    Raises:
        psycopg2.OperationalError: if the database cannot be reached.
        psycopg2.Error: on SQL execution errors.
    """
    conn = _connect_with_retry(POSTGRES_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                logger.info("Enabling pgvector extension…")
                cur.execute(CREATE_EXTENSION_SQL)

                logger.info("Creating chunks table…")
                cur.execute(CREATE_CHUNKS_TABLE_SQL)

                logger.info("Creating HNSW embedding index…")
                cur.execute(CREATE_HNSW_INDEX_SQL)

                logger.info("Creating fiscal_year index…")
                cur.execute(CREATE_FISCAL_YEAR_INDEX_SQL)

                logger.info("Creating source_doc index…")
                cur.execute(CREATE_SOURCE_DOC_INDEX_SQL)

                logger.info("Creating bm25_corpus table…")
                cur.execute(CREATE_BM25_CORPUS_TABLE_SQL)

        logger.info("Database initialisation complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
