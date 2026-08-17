# jet_python_modules/jet/libs/unstructured_lib/jet_examples/auto_partitions/demo_partition_pdf.py
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Base path for example documents relative to this script's location
# Adjust if your environment mounts example-docs differently
EXAMPLE_DOCS_DIR = Path(
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/unstructured/example-docs"
)

INPUT_PDF = "/Users/jethroestrada/Downloads/web_swarm_linkedin_post.pdf"


def _save_elements(elements: list[Element], name: str) -> None:
    """Serialize elements to JSON in OUTPUT_DIR and log summary."""
    output_path = OUTPUT_DIR / f"{name}.json"
    data = [el.to_dict() for el in elements]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(
        "✅ %-30s | %4d elements | saved to %s",
        name,
        len(elements),
        output_path.relative_to(OUTPUT_DIR.parent),
    )


# -----------------------------------------------------------------------------
# Demo Functions
# -----------------------------------------------------------------------------
def demo_fast_strategy() -> None:
    """Fast strategy: direct text extraction via pdfminer without layout model."""
    logger.info("🚀 Running demo_fast_strategy...")
    filename = INPUT_PDF
    elements = partition_pdf(
        filename=filename,
        strategy="fast",
        include_page_breaks=True,
    )
    _save_elements(elements, "fast_strategy")


def demo_hi_res_strategy() -> None:
    """Hi-res strategy: layout detection model + table structure inference."""
    logger.info("🚀 Running demo_hi_res_strategy...")
    filename = INPUT_PDF
    elements = partition_pdf(
        filename=filename,
        strategy="hi_res",
        infer_table_structure=True,
        languages=["eng"],
    )
    _save_elements(elements, "hi_res_strategy")


def demo_ocr_only_strategy() -> None:
    """OCR-only strategy: full page OCR without layout model or pdfminer."""
    logger.info("🚀 Running demo_ocr_only_strategy...")
    filename = INPUT_PDF
    elements = partition_pdf(
        filename=filename,
        strategy="ocr_only",
        languages=["eng"],
    )
    _save_elements(elements, "ocr_only_strategy")


def demo_password_protected() -> None:
    """Handle password-protected PDFs by providing the password parameter."""
    logger.info("🚀 Running demo_password_protected...")
    filename = INPUT_PDF
    elements = partition_pdf(
        filename=filename,
        strategy="fast",
        password="password",
    )
    _save_elements(elements, "password_protected")


def demo_rotated_page() -> None:
    """Hi-res strategy on a 90-degree rotated page to test rotation correction."""
    logger.info("🚀 Running demo_rotated_page...")
    filename = INPUT_PDF
    elements = partition_pdf(
        filename=filename,
        strategy="hi_res",
        languages=["eng"],
    )
    _save_elements(elements, "rotated_page")


def demo_embedded_images() -> None:
    """Extract embedded images and tables as base64 payloads in metadata."""
    logger.info("🚀 Running demo_embedded_images...")
    filename = INPUT_PDF
    elements = partition_pdf(
        filename=filename,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
    )
    _save_elements(elements, "embedded_images")


def demo_complex_vector() -> None:
    """Auto strategy on a complex/vector-heavy PDF that may trigger fallback logic."""
    logger.info("🚀 Running demo_complex_vector...")
    filename = INPUT_PDF
    elements = partition_pdf(
        filename=filename,
        strategy="auto",
    )
    _save_elements(elements, "complex_vector")


def demo_copy_protected() -> None:
    """Hi-res strategy on a copy-protected PDF where text selection is disabled."""
    logger.info("🚀 Running demo_copy_protected...")
    filename = INPUT_PDF
    elements = partition_pdf(
        filename=filename,
        strategy="hi_res",
        languages=["eng"],
    )
    _save_elements(elements, "copy_protected")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
DEMO_FUNCTIONS = [
    demo_fast_strategy,
    demo_hi_res_strategy,
    demo_ocr_only_strategy,
    demo_password_protected,
    demo_rotated_page,
    demo_embedded_images,
    demo_complex_vector,
    demo_copy_protected,
]


def main() -> None:
    """Run all PDF partition demos sequentially."""
    logger.info("=" * 70)
    logger.info("PDF Partition Demos — Output: %s", OUTPUT_DIR)
    logger.info("=" * 70)

    failed: list[str] = []
    for demo_fn in DEMO_FUNCTIONS:
        try:
            demo_fn()
        except Exception as exc:
            logger.error("❌ %s failed: %s", demo_fn.__name__, exc, exc_info=True)
            failed.append(demo_fn.__name__)

    logger.info("=" * 70)
    if failed:
        logger.warning(
            "⚠️  %d/%d demos failed: %s",
            len(failed),
            len(DEMO_FUNCTIONS),
            ", ".join(failed),
        )
    else:
        logger.info("🎉 All %d demos completed successfully!", len(DEMO_FUNCTIONS))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
