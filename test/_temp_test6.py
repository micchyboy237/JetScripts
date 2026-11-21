"""Demonstration of parse_document across diverse PDF types with RAG chunking.

This module exercises the hierarchical RAG chunker against a curated set of PDFs
from the unstructured example-docs directory. Each demo function parses a specific
PDF variant and persists the full parse_document result (elements, chunks,
rag_context, metadata) as JSON under OUTPUT_DIR.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from jet.adapters.unstructured.document_parser import parse_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


INPUT_PDF = "/Users/jethroestrada/Downloads/web_swarm_linkedin_post.pdf"


def _save_result(result: Dict[str, Any], label: str) -> None:
    """Serialize a parse_document result dict to JSON under OUTPUT_DIR.

    Args:
        result: Full dict returned by parse_document.
        label: Human-readable label used as the output filename stem.
    """
    output_path = OUTPUT_DIR / f"{label}.json"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(
            f"Saved result for '{label}' → {output_path.name} | "
            f"elements={result.get('element_count', 0)} | "
            f"chunks={len(result.get('chunks', []))} | "
            f"status={result.get('status', 'unknown')}"
        )
    except Exception as e:
        logger.error(f"Failed to save result for '{label}': {e}")


def _run_parse(filename: str, label: Optional[str] = None, **kwargs: Any) -> None:
    """Locate a PDF, parse it with parse_document, and save the result.

    Args:
        filename: Name of the PDF to find recursively under EXAMPLE_DOCS_DIR.
        label: Output filename stem. Defaults to filename without extension.
        **kwargs: Extra keyword arguments forwarded to parse_document.
    """
    file_path = Path(INPUT_PDF)
    if file_path is None:
        logger.warning(f"File not found, skipping: {filename}")
        return

    display_label = label or file_path.stem
    logger.info(f"Parsing: {display_label}")

    result = parse_document(str(file_path), **kwargs)
    _save_result(result, display_label)


# ---------------------------------------------------------------------------
# Individual demo functions – each covers a distinct PDF parsing scenario
# ---------------------------------------------------------------------------


def demo_pdf_standard_memo() -> None:
    """Parse a simple single-page memo (baseline narrative + header)."""
    _run_parse("fake-memo.pdf", label="01_standard_memo")


def demo_pdf_with_table() -> None:
    """Parse a PDF containing embedded tables (tests atomic chunking)."""
    _run_parse("embedded-images-tables.pdf", label="02_embedded_tables")


def demo_pdf_multi_column() -> None:
    """Parse a multi-column layout PDF (tests reading-order detection)."""
    _run_parse("multi-column-2p.pdf", label="03_multi_column")


def demo_pdf_ocr_text() -> None:
    """Parse a scanned/image-based PDF requiring OCR."""
    _run_parse("pdf-with-ocr-text.pdf", label="04_ocr_text")


def demo_pdf_large_document() -> None:
    """Parse a large multi-page document (tests chunk budget & overlap)."""
    _run_parse("DA-619p.pdf", label="05_large_619p")


def demo_pdf_rotated_page() -> None:
    """Parse a PDF with a 90° rotated page (tests orientation handling)."""
    _run_parse("rotated-page-90.pdf", label="06_rotated_page")


def demo_pdf_list_items() -> None:
    """Parse a PDF heavy with list items (tests ListItem chunking)."""
    _run_parse("list-item-example.pdf", label="07_list_items")


def demo_pdf_korean_tables() -> None:
    """Parse a non-Latin PDF with tables (tests multilingual + atomic)."""
    _run_parse("korean-text-with-tables.pdf", label="08_korean_tables")


def demo_pdf_header_footer() -> None:
    """Parse a PDF with explicit headers/footers (tests section anchoring)."""
    _run_parse("header-test-doc.pdf", label="09_header_footer")


def demo_pdf_single_table() -> None:
    """Parse a PDF that is essentially one large table (atomic_flat strategy)."""
    _run_parse("single_table.pdf", label="10_single_table")


def run_all_demos() -> None:
    """Execute all PDF demo functions sequentially."""
    demos = [
        demo_pdf_standard_memo,
        demo_pdf_with_table,
        demo_pdf_multi_column,
        demo_pdf_ocr_text,
        demo_pdf_large_document,
        demo_pdf_rotated_page,
        demo_pdf_list_items,
        demo_pdf_korean_tables,
        demo_pdf_header_footer,
        demo_pdf_single_table,
    ]

    logger.info(f"Starting parse_document PDF demos. Output: {OUTPUT_DIR}")

    for demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.exception(f"Unhandled error in {demo_func.__name__}: {e}")

    logger.info("All PDF demos completed.")


if __name__ == "__main__":
    run_all_demos()
