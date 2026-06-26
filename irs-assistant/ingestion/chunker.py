"""
Text chunker for IRS document ingestion.

Splits RawChunks (one per article) into token-bounded Chunks suitable for
embedding. Article boundaries are always respected — a new article always
starts a new chunk. Long articles are split at sentence boundaries with
overlap to preserve cross-split context.
"""

import logging
import re
from dataclasses import dataclass

from ingestion.parser import RawChunk

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS: int = 450
DEFAULT_OVERLAP_TOKENS: int = 50
TOKENS_PER_WORD: float = 1.3

SENTENCE_BOUNDARY_RE: re.Pattern = re.compile(r"(?<=\. )(?=[A-Z])")


@dataclass
class Chunk:
    """A token-bounded chunk of text ready for embedding.

    Mirrors all RawChunk fields and adds a sequential chunk_index.

    Args:
        content: Text content of this chunk.
        source_doc: Source document name.
        article: Article identifier, or None.
        section: Section/chapter heading, or None.
        page_number: Starting page in the source PDF, or None.
        fiscal_year: Fiscal year this chunk belongs to.
        chunk_index: Sequential index across all chunks in a single document.
    """

    content: str
    source_doc: str
    article: str | None
    section: str | None
    page_number: int | None
    fiscal_year: int
    chunk_index: int


def _approx_tokens(text: str) -> float:
    """Approximate token count using word count * TOKENS_PER_WORD."""
    return len(text.split()) * TOKENS_PER_WORD


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries ('. ' followed by ASCII uppercase)."""
    return [s for s in SENTENCE_BOUNDARY_RE.split(text) if s.strip()]


def _overlap_prefix(text: str, overlap_tokens: int) -> str:
    """Return the last ~overlap_tokens worth of words from text."""
    word_count = round(overlap_tokens / TOKENS_PER_WORD)
    words = text.split()
    return " ".join(words[-word_count:]) if words else ""


def chunk_raw(
    raw_chunks: list[RawChunk],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split RawChunks into token-bounded Chunks.

    Rules:
    - A short article (≤ max_tokens) becomes exactly one Chunk.
    - A long article is split at sentence boundaries. The last ~overlap_tokens
      words of each sub-chunk are prepended to the next to preserve context.
    - If a single sentence exceeds max_tokens (rare in legal text), it is
      split by word count as a hard fallback.
    - chunk_index is sequential across all chunks produced by this call.
      Call chunk_raw once per document, not once globally across all documents.

    Args:
        raw_chunks: List of per-article RawChunk objects from parse_pdf/parse_html.
        max_tokens: Soft token limit per chunk (approximated as words * 1.3).
        overlap_tokens: Number of tokens to repeat at the start of split continuations.

    Returns:
        Flat list of Chunk objects with sequential chunk_index values.

    Raises:
        ValueError: if raw_chunks is empty (caller should not embed an empty document).
    """
    result: list[Chunk] = []
    chunk_index: int = 0

    def _emit(content: str, raw: RawChunk) -> None:
        nonlocal chunk_index
        result.append(
            Chunk(
                content=content,
                source_doc=raw.source_doc,
                article=raw.article,
                section=raw.section,
                page_number=raw.page_number,
                fiscal_year=raw.fiscal_year,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1

    for raw in raw_chunks:
        if _approx_tokens(raw.content) <= max_tokens:
            _emit(raw.content, raw)
            continue

        sentences = _split_sentences(raw.content)
        current_sentences: list[str] = []
        current_tokens: float = 0.0
        overlap_text: str = ""

        for sentence in sentences:
            sentence_tokens = _approx_tokens(sentence)

            # Flush accumulated sentences before processing this one if it won't fit
            if current_tokens + sentence_tokens > max_tokens and current_sentences:
                chunk_text = " ".join(current_sentences)
                full_text = (
                    (overlap_text + " " + chunk_text).strip()
                    if overlap_text
                    else chunk_text
                )
                _emit(full_text, raw)
                overlap_text = _overlap_prefix(chunk_text, overlap_tokens)
                current_sentences = []
                current_tokens = 0.0

            # Hard-split a sentence that exceeds the limit on its own
            if sentence_tokens > max_tokens:
                words = sentence.split()
                step = round(max_tokens / TOKENS_PER_WORD)
                while words:
                    segment = " ".join(words[:step])
                    full_segment = (
                        (overlap_text + " " + segment).strip()
                        if overlap_text
                        else segment
                    )
                    _emit(full_segment, raw)
                    overlap_text = _overlap_prefix(segment, overlap_tokens)
                    words = words[step:]
                continue

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            full_text = (
                (overlap_text + " " + chunk_text).strip()
                if overlap_text
                else chunk_text
            )
            _emit(full_text, raw)

    logger.info("Produced %d chunks from %d raw chunks.", len(result), len(raw_chunks))
    return result


def validate_chunks(
    chunks: list[Chunk],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> None:
    """Validate that all chunks meet quality requirements before embedding.

    Args:
        chunks: List of Chunk objects to validate.
        max_tokens: The max_tokens value used during chunking (for the 1.5x ceiling check).

    Raises:
        ValueError: with a descriptive message and chunk_index if any chunk fails validation.
    """
    for chunk in chunks:
        if not chunk.content or not chunk.content.strip():
            raise ValueError(
                f"Chunk at index {chunk.chunk_index} has empty content."
            )
        if _approx_tokens(chunk.content) > max_tokens * 1.5:
            raise ValueError(
                f"Chunk at index {chunk.chunk_index} exceeds {max_tokens * 1.5:.0f} "
                f"tokens (approx {_approx_tokens(chunk.content):.0f})."
            )
        if not chunk.source_doc:
            raise ValueError(
                f"Chunk at index {chunk.chunk_index} has no source_doc."
            )
        if not chunk.fiscal_year:
            raise ValueError(
                f"Chunk at index {chunk.chunk_index} has no fiscal_year."
            )
