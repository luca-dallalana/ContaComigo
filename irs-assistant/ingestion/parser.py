"""
Document parsers for PDF and HTML IRS source documents.

Each parser produces one RawChunk per logical article. Articles can span
multiple PDF pages; the parser merges them. The page_number field records
where the article starts. The chunker is responsible for splitting long
articles — never the parser.
"""

import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import pypdf

from ingestion.downloader import DocumentSource

logger = logging.getLogger(__name__)

ARTICLE_REGEX: re.Pattern = re.compile(r"Artigo\s+\d+[\.º°]\s*[A-Z\-]*")
SECTION_REGEX: re.Pattern = re.compile(
    r"(?:CAPÍTULO|SECÇÃO|TÍTULO)\s+[IVXivx\d]+"
)
_MONTH_PT: str = (
    r"(?:JANEIRO|FEVEREIRO|MARÇO|ABRIL|MAIO|JUNHO"
    r"|JULHO|AGOSTO|SETEMBRO|OUTUBRO|NOVEMBRO|DEZEMBRO)"
)
_YEAR_OPT: str = r"(?:\s+DE\s+\d{4})?"
DEADLINE_REGEX: re.Pattern = re.compile(
    r"("
    + rf"ATÉ\s+(?:AO\s+(?:DIA\s+\d+|FINAL\s+DE\s+{_MONTH_PT}{_YEAR_OPT})|\d+\s+DE\s+{_MONTH_PT}{_YEAR_OPT})"
    + r"|"
    + rf"DE\s+(?:\d+\s+A\s+\d+\s+DE\s+{_MONTH_PT}{_YEAR_OPT}|{_MONTH_PT}\s+A\s+{_MONTH_PT}{_YEAR_OPT})"
    + r")"
)
MIN_CHUNK_CHARS: int = 20

SKIP_TAGS: frozenset = frozenset(
    {"nav", "header", "footer", "script", "style", "noscript", "aside"}
)


@dataclass
class RawChunk:
    """A single article-sized unit of text extracted from a source document.

    Args:
        content: Raw text of the article or section.
        source_doc: Human-readable document name (matches DocumentSource.source_doc_name).
        article: Article identifier string, e.g. "Artigo 12.º", or None for preamble text.
        section: Chapter/section heading if detectable, otherwise None.
        page_number: 1-indexed starting page in the PDF; None for HTML sources.
        fiscal_year: Fiscal year the document covers.
    """

    content: str
    source_doc: str
    article: str | None
    section: str | None
    page_number: int | None
    fiscal_year: int


class _IRSHTMLParser(HTMLParser):
    """Extracts visible text from IRS HTML pages, skipping navigation/boilerplate tags."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth: int = 0
        self._parts: list[str] = []
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._tag_stack.append(tag)
        if tag in SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
        if tag in SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        """Return all extracted visible text joined by newlines."""
        return "\n".join(self._parts)


def parse_pdf(filepath: Path, source: DocumentSource) -> list[RawChunk]:
    """Parse a PDF file into one RawChunk per logical article.

    Articles may span multiple pages; text is accumulated until the next
    article boundary is detected. Fragments shorter than MIN_CHUNK_CHARS
    are discarded.

    Args:
        filepath: Path to the PDF file.
        source: DocumentSource metadata for this file.

    Returns:
        List of RawChunk objects, one per detected article (plus a preamble
        chunk if text precedes the first article heading).

    Raises:
        pypdf.errors.PdfReadError: if the file is not a valid PDF.
        OSError: if the file cannot be opened.
    """
    reader = pypdf.PdfReader(str(filepath))
    chunks: list[RawChunk] = []

    current_article: str | None = None
    current_section: str | None = None
    current_start_page: int | None = None
    buffer: list[str] = []

    def _flush() -> None:
        text = " ".join(buffer).strip()
        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append(
                RawChunk(
                    content=text,
                    source_doc=source.source_doc_name,
                    article=current_article,
                    section=current_section,
                    page_number=current_start_page,
                    fiscal_year=source.fiscal_year,
                )
            )
        buffer.clear()

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        section_match = SECTION_REGEX.search(text)
        if section_match:
            current_section = section_match.group().strip()

        last_end = 0
        for match in ARTICLE_REGEX.finditer(text):
            pre_text = text[last_end : match.start()]
            if pre_text.strip():
                buffer.append(pre_text)

            _flush()

            current_article = match.group().strip()
            current_start_page = page_num
            last_end = match.end()

        remaining = text[last_end:]
        if remaining.strip():
            buffer.append(remaining)

    _flush()

    logger.info("Parsed %d raw chunks from '%s'.", len(chunks), filepath.name)
    return chunks


def parse_calendar_pdf(filepath: Path, source: DocumentSource) -> list[RawChunk]:
    """Parse a deadline/calendar PDF into one RawChunk per deadline entry.

    Splits on "ATÉ [date]" patterns rather than article headings. Each chunk
    is self-contained: the deadline date is stored in the article field so
    format_chunks displays it as the label (e.g. "ATÉ 30 DE JUNHO DE 2026").

    Args:
        filepath: Path to the PDF file.
        source: DocumentSource metadata for this file.

    Returns:
        List of RawChunk objects, one per detected deadline entry.
    """
    reader = pypdf.PdfReader(str(filepath))
    full_text = " ".join((page.extract_text() or "") for page in reader.pages)

    parts = re.split(DEADLINE_REGEX, full_text)

    chunks: list[RawChunk] = []
    preamble = parts[0].strip()
    if len(preamble) >= MIN_CHUNK_CHARS:
        chunks.append(
            RawChunk(
                content=preamble,
                source_doc=source.source_doc_name,
                article=None,
                section=None,
                page_number=None,
                fiscal_year=source.fiscal_year,
            )
        )

    i = 1
    while i + 1 < len(parts):
        deadline_header = parts[i].strip()
        deadline_body = parts[i + 1].strip()
        # Append the next date header as a look-ahead so the chunk contains both
        # the action body and the deadline date that closes it. This is necessary
        # because AT documents list the action first, then the date ("ENTREGUE O IRS
        # ... DE ABRIL A JUNHO DE 2026"), so without look-ahead the date would be
        # isolated in the following chunk's header with no action context.
        next_date = parts[i + 2].strip() if i + 2 < len(parts) else ""
        if next_date:
            # AT calendar documents list the action body before its closing date.
            # Put the closing date first so the chunk reads naturally: when → what.
            combined = f"{next_date}\n{deadline_body}"
            article_label = next_date
        else:
            combined = f"{deadline_header}\n{deadline_body}"
            article_label = deadline_header
        combined = combined.strip()
        if len(combined) >= MIN_CHUNK_CHARS:
            chunks.append(
                RawChunk(
                    content=combined,
                    source_doc=source.source_doc_name,
                    article=article_label,
                    section=None,
                    page_number=None,
                    fiscal_year=source.fiscal_year,
                )
            )
        i += 2

    logger.info("Parsed %d raw chunks from '%s'.", len(chunks), filepath.name)
    return chunks


def parse_html(filepath: Path, source: DocumentSource) -> list[RawChunk]:
    """Parse an HTML file into one RawChunk per logical article.

    Uses Python's built-in html.parser — no third-party HTML library.
    Navigation, headers, footers, scripts, and styles are stripped.

    Args:
        filepath: Path to the HTML file.
        source: DocumentSource metadata for this file.

    Returns:
        List of RawChunk objects, one per detected article.

    Raises:
        OSError: if the file cannot be read.
        UnicodeDecodeError: if the file encoding cannot be determined
            (errors are replaced, so this should not propagate).
    """
    raw_html = filepath.read_text(encoding="utf-8", errors="replace")
    parser = _IRSHTMLParser()
    parser.feed(raw_html)
    text = parser.get_text()

    parts = re.split(r"(Artigo\s+\d+[\.º°]\s*[A-Z\-]*)", text)

    chunks: list[RawChunk] = []
    preamble = parts[0].strip()
    if len(preamble) >= MIN_CHUNK_CHARS:
        chunks.append(
            RawChunk(
                content=preamble,
                source_doc=source.source_doc_name,
                article=None,
                section=None,
                page_number=None,
                fiscal_year=source.fiscal_year,
            )
        )

    i = 1
    while i + 1 < len(parts):
        article_header = parts[i].strip()
        article_body = parts[i + 1].strip()
        combined = f"{article_header}\n{article_body}".strip()
        if len(combined) >= MIN_CHUNK_CHARS:
            chunks.append(
                RawChunk(
                    content=combined,
                    source_doc=source.source_doc_name,
                    article=article_header,
                    section=None,
                    page_number=None,
                    fiscal_year=source.fiscal_year,
                )
            )
        i += 2

    logger.info("Parsed %d raw chunks from '%s'.", len(chunks), filepath.name)
    return chunks
