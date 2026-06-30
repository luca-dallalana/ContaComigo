"""
Reciprocal Rank Fusion for combining vector and BM25 search results.

RRF formula (Cormack et al. 2009): score = Σ weight / (k + rank)
where rank is 1-indexed. A chunk appearing in both result lists gets a
combined score — it is never duplicated in the output.
"""

import dataclasses
import logging

from retrieval.vector_search import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_RRF_K: int = 60


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    k: int = DEFAULT_RRF_K,
) -> list[SearchResult]:
    """Combine vector and BM25 search results using Reciprocal Rank Fusion.

    RRF assigns each result a score of weight / (k + rank) where rank is
    1-indexed. Results appearing in both lists receive the sum of both
    contributions. The output is sorted by combined score descending.

    This function does not trim to TOP_K — the caller applies that limit
    after fusion using the TOP_K env variable.

    Args:
        vector_results: Results from vector_search, sorted by score descending.
        bm25_results: Results from bm25_search, sorted by score descending.
        vector_weight: Weight applied to vector search ranks (default 0.7).
        bm25_weight: Weight applied to BM25 ranks (default 0.3).
        k: RRF smoothing constant (default 60, from the original paper).

    Returns:
        Deduplicated list of SearchResult sorted by RRF score descending.
        The .score field on each result is the accumulated RRF score,
        replacing the original vector/BM25 score. Input objects are not
        mutated.

    Raises:
        Nothing — empty input lists are handled gracefully.
    """
    rrf_scores: dict[int, float] = {}
    result_by_id: dict[int, SearchResult] = {}

    for rank, result in enumerate(vector_results):
        rrf_scores[result.chunk_id] = (
            rrf_scores.get(result.chunk_id, 0.0)
            + vector_weight / (k + rank + 1)
        )
        result_by_id[result.chunk_id] = result

    for rank, result in enumerate(bm25_results):
        rrf_scores[result.chunk_id] = (
            rrf_scores.get(result.chunk_id, 0.0)
            + bm25_weight / (k + rank + 1)
        )
        result_by_id[result.chunk_id] = result

    sorted_ids = sorted(
        rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True
    )

    return [
        dataclasses.replace(result_by_id[cid], score=rrf_scores[cid])
        for cid in sorted_ids
    ]
