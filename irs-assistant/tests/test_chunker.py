import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.chunker import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    Chunk,
    _approx_tokens,
    _overlap_prefix,
    _split_sentences,
    chunk_raw,
    validate_chunks,
)
from ingestion.parser import RawChunk


def _raw(content: str, article: str | None = "Artigo 1.º", fiscal_year: int = 2024) -> RawChunk:
    return RawChunk(
        content=content,
        source_doc="Test Doc",
        article=article,
        section=None,
        page_number=1,
        fiscal_year=fiscal_year,
    )


def _chunk(content: str, source_doc: str = "Test Doc", fiscal_year: int = 2024) -> Chunk:
    return Chunk(
        content=content,
        source_doc=source_doc,
        article="Artigo 1.º",
        section=None,
        page_number=1,
        fiscal_year=fiscal_year,
        chunk_index=0,
    )


# --- _approx_tokens ---

def test_approx_tokens_two_words():
    assert _approx_tokens("hello world") == pytest.approx(2 * 1.3)


def test_approx_tokens_empty():
    assert _approx_tokens("") == 0.0


def test_approx_tokens_single_word():
    assert _approx_tokens("word") == pytest.approx(1.3)


# --- _split_sentences ---

def test_split_sentences_two():
    result = _split_sentences("First sentence. Second sentence.")
    assert len(result) == 2
    assert result[0].strip() == "First sentence."
    assert result[1].strip() == "Second sentence."


def test_split_sentences_no_split():
    result = _split_sentences("No split here")
    assert result == ["No split here"]


def test_split_sentences_filters_empty():
    result = _split_sentences("   ")
    assert result == []


# --- _overlap_prefix ---

def test_overlap_prefix_returns_last_words():
    result = _overlap_prefix("one two three four five", 4)
    words = result.split()
    assert len(words) <= 4


def test_overlap_prefix_empty_text():
    assert _overlap_prefix("", 50) == ""


# --- chunk_raw ---

def test_chunk_raw_short_article_is_one_chunk():
    short = "word " * 10
    chunks = chunk_raw([_raw(short)])
    assert len(chunks) == 1
    assert chunks[0].content == short


def test_chunk_raw_long_article_splits():
    long = ("word " * 50 + ". ") * 20
    chunks = chunk_raw([_raw(long)])
    assert len(chunks) > 1
    for c in chunks:
        assert _approx_tokens(c.content) <= DEFAULT_MAX_TOKENS * 1.5


def test_chunk_raw_overlap_in_second_chunk():
    sentence = "word " * 40
    long = (sentence + ". ") * 3
    chunks = chunk_raw([_raw(long)])
    if len(chunks) >= 2:
        last_words_of_first = chunks[0].content.split()[-3:]
        assert any(w in chunks[1].content for w in last_words_of_first)


def test_chunk_raw_sequential_index():
    raw1 = _raw("short text one")
    raw2 = _raw("short text two")
    chunks = chunk_raw([raw1, raw2])
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_raw_preserves_metadata():
    raw = RawChunk(
        content="short content",
        source_doc="My Doc",
        article="Artigo 5.º",
        section="Capítulo I",
        page_number=3,
        fiscal_year=2025,
    )
    chunks = chunk_raw([raw])
    assert chunks[0].source_doc == "My Doc"
    assert chunks[0].article == "Artigo 5.º"
    assert chunks[0].fiscal_year == 2025
    assert chunks[0].page_number == 3


def test_chunk_raw_empty_input_returns_empty():
    assert chunk_raw([]) == []


# --- validate_chunks ---

def test_validate_chunks_valid_passes():
    validate_chunks([_chunk("Some valid content.")])


def test_validate_chunks_empty_content_raises():
    with pytest.raises(ValueError, match="empty content"):
        validate_chunks([_chunk("")])


def test_validate_chunks_whitespace_content_raises():
    with pytest.raises(ValueError, match="empty content"):
        validate_chunks([_chunk("   ")])


def test_validate_chunks_over_token_limit_raises():
    huge = "word " * int(DEFAULT_MAX_TOKENS * 1.6 / 1.3)
    with pytest.raises(ValueError, match="tokens"):
        validate_chunks([_chunk(huge)])


def test_validate_chunks_missing_source_doc_raises():
    c = _chunk("some content")
    c.source_doc = ""
    with pytest.raises(ValueError, match="source_doc"):
        validate_chunks([c])


def test_validate_chunks_missing_fiscal_year_raises():
    c = _chunk("some content")
    c.fiscal_year = 0
    with pytest.raises(ValueError, match="fiscal_year"):
        validate_chunks([c])
