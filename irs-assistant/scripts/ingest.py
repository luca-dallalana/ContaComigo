"""
Orchestration script for the IRS document ingestion pipeline.

Runs the full pipeline: download → parse → chunk → validate → embed → store.
Accepts --fiscal-year and --source flags for partial ingestion. Prompts for
confirmation before re-ingesting a source that already has chunks in the DB.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --fiscal-year 2024
    python scripts/ingest.py --source "Guia do IRS 2024"
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import psycopg2
import psycopg2.extensions
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.chunker import chunk_raw, validate_chunks
from ingestion.downloader import (
    DOCUMENT_SOURCES,
    DocType,
    DocumentSource,
    download_documents,
)
from ingestion.embedder import embed_chunks, store_chunks
from ingestion.parser import parse_calendar_pdf, parse_html, parse_pdf

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

POSTGRES_URL: str = os.environ["POSTGRES_URL"]
DATA_DIR: Path = Path(__file__).parent.parent / "data"

COUNT_CHUNKS_SQL = (
    "SELECT COUNT(*) FROM chunks WHERE source_doc = %s AND fiscal_year = %s"
)
DELETE_BM25_SQL = """
DELETE FROM bm25_corpus
WHERE chunk_id IN (
    SELECT id FROM chunks WHERE source_doc = %s AND fiscal_year = %s
)
"""
DELETE_CHUNKS_SQL = (
    "DELETE FROM chunks WHERE source_doc = %s AND fiscal_year = %s"
)


def _check_existing(
    conn: psycopg2.extensions.connection,
    source_doc: str,
    fiscal_year: int,
) -> int:
    """Return the number of existing chunks for a (source_doc, fiscal_year) pair.

    Args:
        conn: Open psycopg2 connection.
        source_doc: Source document name to check.
        fiscal_year: Fiscal year to check.

    Returns:
        Row count (0 if not yet ingested).

    Raises:
        psycopg2.Error: on database errors.
    """
    with conn.cursor() as cur:
        cur.execute(COUNT_CHUNKS_SQL, (source_doc, fiscal_year))
        return cur.fetchone()[0]


def _delete_existing(
    conn: psycopg2.extensions.connection,
    source_doc: str,
    fiscal_year: int,
) -> None:
    """Delete all chunks and their BM25 tokens for a (source_doc, fiscal_year) pair.

    bm25_corpus rows are deleted first because the FK (chunk_id → chunks.id)
    has no ON DELETE CASCADE.

    Args:
        conn: Open psycopg2 connection.
        source_doc: Source document name to delete.
        fiscal_year: Fiscal year to delete.

    Raises:
        psycopg2.Error: on database errors (transaction is rolled back).
    """
    with conn:
        with conn.cursor() as cur:
            cur.execute(DELETE_BM25_SQL, (source_doc, fiscal_year))
            cur.execute(DELETE_CHUNKS_SQL, (source_doc, fiscal_year))
    logger.info("Deleted existing data for '%s' (%d).", source_doc, fiscal_year)


def main() -> None:
    """Run the full ingestion pipeline.

    Raises:
        SystemExit: on unrecoverable configuration errors.
    """
    arg_parser = argparse.ArgumentParser(
        description="Ingest IRS documents into pgvector."
    )
    arg_parser.add_argument("--fiscal-year", type=int, help="Filter by fiscal year.")
    arg_parser.add_argument("--source", type=str, help="Filter by source_doc_name.")
    args = arg_parser.parse_args()

    t0 = time.time()

    sources = list(DOCUMENT_SOURCES)
    if args.fiscal_year:
        sources = [s for s in sources if s.fiscal_year == args.fiscal_year]
    if args.source:
        sources = [s for s in sources if s.source_doc_name == args.source]

    if not sources:
        print("No sources matched the provided filters.")
        return

    conn = psycopg2.connect(POSTGRES_URL)
    sources_to_process: list[DocumentSource] = []
    try:
        for source in sources:
            count = _check_existing(conn, source.source_doc_name, source.fiscal_year)
            if count > 0:
                answer = input(
                    f"Found {count} existing chunks for '{source.source_doc_name}' "
                    f"(fiscal_year={source.fiscal_year}). Re-ingest? [y/N] "
                )
                if answer.strip().lower() != "y":
                    print(f"Skipping '{source.source_doc_name}'.")
                    continue
                _delete_existing(conn, source.source_doc_name, source.fiscal_year)
            sources_to_process.append(source)
    finally:
        conn.close()

    if not sources_to_process:
        print("Nothing to ingest.")
        return

    downloaded = download_documents(sources_to_process, output_dir=DATA_DIR)

    if not downloaded:
        print("No documents were downloaded successfully. Check logs for errors.")
        return

    total_docs = 0
    total_chunks = 0

    conn = psycopg2.connect(POSTGRES_URL)
    try:
        for source, filepath in downloaded:
            print(f"\nProcessing: {source.source_doc_name}")

            if source.doc_type == DocType.PDF:
                raw_chunks = parse_pdf(filepath, source)
            elif source.doc_type == DocType.CALENDAR:
                raw_chunks = parse_calendar_pdf(filepath, source)
            else:
                raw_chunks = parse_html(filepath, source)

            print(f"  Parsed {len(raw_chunks)} raw article chunks.")

            chunks = chunk_raw(raw_chunks)
            print(f"  Split into {len(chunks)} chunks.")

            validate_chunks(chunks)

            chunks_with_emb = embed_chunks(chunks)

            n = store_chunks(chunks_with_emb, conn)
            total_chunks += n
            total_docs += 1
            print(f"  Stored {n} chunks for '{source.source_doc_name}'.")
    finally:
        conn.close()

    elapsed = time.time() - t0
    print(
        f"\nDone. {total_docs} document(s), {total_chunks} chunk(s) stored "
        f"in {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
