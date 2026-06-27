import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.prompt import SECTION_FALLBACK, build_prompt, format_chunks
from retrieval.vector_search import SearchResult


def _sr(
    content: str,
    source_doc: str = "Código do IRS (CIRS)",
    article: str | None = "Artigo 1.º",
) -> SearchResult:
    return SearchResult(
        chunk_id=1,
        content=content,
        source_doc=source_doc,
        article=article,
        page_number=None,
        fiscal_year=2024,
        score=0.9,
    )


# --- format_chunks ---

def test_format_chunks_empty_returns_empty_string():
    assert format_chunks([]) == ""


def test_format_chunks_none_article_uses_fallback():
    result = format_chunks([_sr("some text", article=None)])
    assert SECTION_FALLBACK in result


def test_format_chunks_single_chunk_structure():
    result = format_chunks([_sr("my content", source_doc="My Doc", article="Artigo 3.º")])
    assert "[1]" in result
    assert "My Doc" in result
    assert "Artigo 3.º" in result
    assert "my content" in result
    assert "---" in result


def test_format_chunks_multiple_numbered():
    chunks = [_sr("content one"), _sr("content two")]
    result = format_chunks(chunks)
    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" not in result


def test_format_chunks_preserves_content():
    result = format_chunks([_sr("exact text here")])
    assert "exact text here" in result


# --- build_prompt ---

def test_build_prompt_contains_question():
    prompt = build_prompt("Qual é a taxa?", [_sr("some chunk")])
    assert "Qual é a taxa?" in prompt


def test_build_prompt_contains_chunk_content():
    prompt = build_prompt("question", [_sr("chunk abc")])
    assert "chunk abc" in prompt


def test_build_prompt_contains_disclaimer_url():
    prompt = build_prompt("question", [_sr("content")])
    assert "portaldasfinancas.gov.pt" in prompt


def test_build_prompt_empty_chunks():
    prompt = build_prompt("question", [])
    assert "question" in prompt
    assert "portaldasfinancas.gov.pt" in prompt
