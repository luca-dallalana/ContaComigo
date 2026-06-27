import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from retrieval.vector_search import SearchResult


def _sr(chunk_id: int, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content=f"content {chunk_id}",
        source_doc="Test Doc",
        article=None,
        page_number=None,
        fiscal_year=2024,
        score=score,
    )


# --- empty inputs ---

def test_rrf_both_empty():
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_vector_only():
    results = reciprocal_rank_fusion([_sr(1), _sr(2), _sr(3)], [])
    assert len(results) == 3
    ids = [r.chunk_id for r in results]
    assert ids == sorted(ids, key=lambda cid: -results[ids.index(cid)].score)


def test_rrf_bm25_only():
    results = reciprocal_rank_fusion([], [_sr(1), _sr(2)])
    assert len(results) == 2


# --- overlap and deduplication ---

def test_rrf_overlap_chunk_ranked_higher():
    vector = [_sr(1), _sr(2)]
    bm25 = [_sr(1), _sr(3)]
    results = reciprocal_rank_fusion(vector, bm25)
    ids = [r.chunk_id for r in results]
    assert ids[0] == 1


def test_rrf_deduplication():
    vector = [_sr(1)]
    bm25 = [_sr(1)]
    results = reciprocal_rank_fusion(vector, bm25)
    assert len(results) == 1


def test_rrf_all_unique_chunks_present():
    vector = [_sr(1), _sr(2)]
    bm25 = [_sr(3), _sr(4)]
    results = reciprocal_rank_fusion(vector, bm25)
    assert {r.chunk_id for r in results} == {1, 2, 3, 4}


# --- score correctness ---

def test_rrf_formula_single_vector_result():
    results = reciprocal_rank_fusion([_sr(1)], [], vector_weight=1.0, bm25_weight=0.0)
    expected = 1.0 / (DEFAULT_RRF_K + 0 + 1)
    assert results[0].score == pytest.approx(expected)


def test_rrf_weight_respected():
    vector = [_sr(10)]
    bm25 = [_sr(20)]
    results = reciprocal_rank_fusion(vector, bm25, vector_weight=0.0, bm25_weight=1.0)
    assert results[0].chunk_id == 20


# --- no mutation ---

def test_rrf_does_not_mutate_inputs():
    sr = _sr(1, score=0.99)
    reciprocal_rank_fusion([sr], [])
    assert sr.score == 0.99


# --- ordering ---

def test_rrf_output_sorted_descending():
    vector = [_sr(1), _sr(2), _sr(3)]
    bm25 = [_sr(2), _sr(3), _sr(1)]
    results = reciprocal_rank_fusion(vector, bm25)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
