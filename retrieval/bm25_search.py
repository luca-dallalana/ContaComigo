"""
BM25 keyword search over the bm25_corpus table.

The BM25 index is built once at application startup from the pre-tokenised
corpus stored in bm25_corpus and held in memory for the lifetime of the
process. Per-query cost is a numpy vector score + a DB fetch for metadata.
"""

import logging
import re

import numpy as np
import psycopg2.extensions
from rank_bm25 import BM25Okapi

from retrieval.vector_search import SearchResult

logger = logging.getLogger(__name__)

TOKEN_SPLIT_RE: re.Pattern = re.compile(r"\W+")

LOAD_CORPUS_SQL: str = "SELECT chunk_id, tokens FROM bm25_corpus ORDER BY id"
FETCH_CHUNKS_SQL: str = (
    "SELECT id, content, source_doc, article, page_number, fiscal_year "
    "FROM chunks WHERE id = ANY(%s)"
)


def build_bm25_index(
    conn: psycopg2.extensions.connection,
) -> tuple[BM25Okapi, list[int]]:
    """Load the tokenised corpus from the database and build a BM25 index.

    The returned index and chunk_ids list are positionally aligned:
    chunk_ids[i] is the database primary key for the document at position i
    in the BM25 index. This alignment is guaranteed by ORDER BY id.

    Call this once at application startup and cache the result. Rebuilding
    per query would scan the full bm25_corpus table on every request.

    Args:
        conn: Open psycopg2 connection. Caller owns lifecycle.

    Returns:
        Tuple of (BM25Okapi index, list of chunk_ids in corpus order).

    Raises:
        psycopg2.Error: on database errors.
    """
    with conn.cursor() as cur:
        cur.execute(LOAD_CORPUS_SQL)
        rows = cur.fetchall()

    chunk_ids: list[int] = [row[0] for row in rows]
    tokenised_corpus: list[list[str]] = [row[1].split() for row in rows]

    index = BM25Okapi(tokenised_corpus)
    logger.info("Built BM25 index over %d documents.", len(chunk_ids))
    return index, chunk_ids


def bm25_search(
    query: str,
    index: BM25Okapi,
    chunk_ids: list[int],
    conn: psycopg2.extensions.connection,
    top_k: int = 10,
) -> list[SearchResult]:
    """Search the corpus using BM25 keyword matching.

    Tokenises the query the same way the corpus was tokenised at ingestion
    time (lowercase, split on non-word characters). Scores are normalised
    to [0, 1] by dividing by the maximum score in the result set.

    Args:
        query: Raw query string in Portuguese.
        index: BM25Okapi index from build_bm25_index.
        chunk_ids: Ordered list of chunk IDs aligned with the index.
        conn: Open psycopg2 connection. Caller owns lifecycle.
        top_k: Maximum number of results to return.

    Returns:
        List of SearchResult with normalised BM25 scores, sorted descending.
        Returns an empty list if no query token matches any document.

    Raises:
        psycopg2.Error: on database errors.
    """
    query_tokens: list[str] = [
        t for t in TOKEN_SPLIT_RE.split(query.lower()) if t
    ]

    scores: np.ndarray = index.get_scores(query_tokens)
    sorted_indices: np.ndarray = np.argsort(scores)[::-1][:top_k]

    top_ids: list[int] = [
        int(chunk_ids[i]) for i in sorted_indices if scores[i] > 0
    ]
    top_scores: list[float] = [
        float(scores[i]) for i in sorted_indices if scores[i] > 0
    ]

    if not top_ids:
        return []

    max_score: float = max(top_scores)
    normalised: list[float] = [s / max_score for s in top_scores]

    with conn.cursor() as cur:
        cur.execute(FETCH_CHUNKS_SQL, (top_ids,))
        rows = cur.fetchall()

    row_by_id: dict[int, tuple] = {row[0]: row for row in rows}

    results: list[SearchResult] = []
    for chunk_id, norm_score in zip(top_ids, normalised):
        if chunk_id not in row_by_id:
            logger.warning("chunk_id %d found in BM25 index but missing from chunks table.", chunk_id)
            continue
        row = row_by_id[chunk_id]
        results.append(
            SearchResult(
                chunk_id=row[0],
                content=row[1],
                source_doc=row[2],
                article=row[3],
                page_number=row[4],
                fiscal_year=row[5],
                score=norm_score,
            )
        )
    return results
