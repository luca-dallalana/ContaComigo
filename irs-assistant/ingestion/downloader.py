"""
Document downloader for IRS source documents.

Downloads PDF and HTML documents from official Portuguese tax authority sources,
skipping already-downloaded files and detecting JS-rendered HTML pages.

Raises:
    httpx.HTTPStatusError: re-raised after logging if all retries fail (not caught here).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

HTTP_TIMEOUT: float = 30.0
MIN_HTML_SIZE_BYTES: int = 5_000
RAW_SUBDIR: str = "raw"
USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class DocType(Enum):
    PDF = "pdf"
    HTML = "html"


@dataclass
class DocumentSource:
    """Describes a single IRS source document to download and ingest.

    Args:
        url: Full URL of the document.
        filename: Local filename to save under data/raw/.
        source_doc_name: Human-readable name used as the source_doc field in chunks.
        fiscal_year: The fiscal year this document applies to.
        doc_type: Whether the document is a PDF or HTML page.
    """

    url: str
    filename: str
    source_doc_name: str
    fiscal_year: int
    doc_type: DocType


DOCUMENT_SOURCES: list[DocumentSource] = [
    DocumentSource(
        url="https://info.portaldasfinancas.gov.pt/pt/informacao_fiscal/codigos_tributarios/Cod_download/Documents/CIRS.pdf",
        filename="cirs_2024.pdf",
        source_doc_name="Código do IRS (CIRS)",
        fiscal_year=2024,
        doc_type=DocType.PDF,
    ),
    DocumentSource(
        url="https://info.portaldasfinancas.gov.pt/pt/apoio_contribuinte/Folhetos_informativos/Documents/IRS_deducoes_2024.pdf",
        filename="guia_deducoes_irs_2024.pdf",
        source_doc_name="Guia de Deduções IRS 2024",
        fiscal_year=2024,
        doc_type=DocType.PDF,
    ),
    # High priority: deadlines and tax calendar
    DocumentSource(
        url="https://info.portaldasfinancas.gov.pt/pt/apoio_contribuinte/Folhetos_informativos/Documents/IRS_2025_Principais_prazos_2026.pdf",
        filename="irs_prazos_2025.pdf",
        source_doc_name="Principais Prazos IRS 2025",
        fiscal_year=2025,
        doc_type=DocType.PDF,
    ),
    DocumentSource(
        url="https://info.portaldasfinancas.gov.pt/pt/apoio_contribuinte/calendario_fiscal/Documents/Obrigacoes_declarativas.pdf",
        filename="agenda_fiscal_2025_declarativa.pdf",
        source_doc_name="Agenda Fiscal 2025 - Obrigações Declarativas",
        fiscal_year=2025,
        doc_type=DocType.PDF,
    ),
    # Medium priority: tax benefits statute and administrative guidance
    DocumentSource(
        url="https://info.portaldasfinancas.gov.pt/pt/informacao_fiscal/codigos_tributarios/Cod_download/Documents/EBF.pdf",
        filename="ebf_2024.pdf",
        source_doc_name="Estatuto dos Benefícios Fiscais (EBF)",
        fiscal_year=2024,
        doc_type=DocType.PDF,
    ),
    DocumentSource(
        url="https://info.portaldasfinancas.gov.pt/pt/informacao_fiscal/legislacao/instrucoes_administrativas/Documents/Oficio_circulado_20269_2024.pdf",
        filename="oficio_circulado_20269_2024.pdf",
        source_doc_name="Ofício Circulado 20269/2024 (IRS)",
        fiscal_year=2024,
        doc_type=DocType.PDF,
    ),
]


def download_documents(
    sources: list[DocumentSource],
    output_dir: Path,
) -> list[tuple[DocumentSource, Path]]:
    """Download source documents to output_dir/raw/, skipping existing files.

    For HTML sources, performs a JS-render check: if the downloaded file is
    smaller than MIN_HTML_SIZE_BYTES, the page is almost certainly JS-rendered
    and the file is deleted rather than stored.

    Args:
        sources: List of DocumentSource entries to download.
        output_dir: Root directory under which raw/ will be created.

    Returns:
        List of (source, filepath) tuples for every successfully downloaded file.
        Sources that fail or are skipped due to JS-render detection are excluded.

    Raises:
        OSError: if output_dir cannot be created.
    """
    raw_dir = output_dir / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[DocumentSource, Path]] = []

    for source in sources:
        dest = raw_dir / source.filename

        if dest.exists():
            logger.info("Skipping '%s', already downloaded.", source.filename)
            results.append((source, dest))
            continue

        try:
            with httpx.Client(
                timeout=HTTP_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            ) as client:
                response = client.get(source.url)
                response.raise_for_status()

            dest.write_bytes(response.content)
            size = dest.stat().st_size

            if source.doc_type == DocType.HTML and size < MIN_HTML_SIZE_BYTES:
                logger.warning(
                    "Downloaded HTML for '%s' is %d bytes — page appears JS-rendered "
                    "(requires a headless browser). Skipping.",
                    source.source_doc_name,
                    size,
                )
                dest.unlink()
                continue

            logger.info("Downloaded '%s' (%d bytes).", source.filename, size)
            results.append((source, dest))

        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP %d downloading '%s' from %s. Skipping.",
                exc.response.status_code,
                source.filename,
                source.url,
            )
        except httpx.RequestError as exc:
            logger.error(
                "Request error downloading '%s': %s. Skipping.",
                source.filename,
                exc,
            )

    return results
