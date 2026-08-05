# === Logging configuration + safe imports ===
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "/Users/jethroestrada/Desktop/External_Projects/"
            "Jet_Projects/JetScripts/test/__sample.log",
            mode="a",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("doc_parser")

# Validate critical dependencies at import time
try:
    # Single unified entry point — replaces all type-specific partitioners
    from unstructured.documents.elements import Text
    from unstructured.partition.auto import partition

    logger.info("✅ unstructured.partition.auto imported successfully")
except ImportError as e:
    logger.critical(f"❌ Missing dependency: {e}")
    logger.critical("   Fix: pip install 'unstructured[all-docs]'")
    sys.exit(1)

from pathlib import Path
from typing import List, Literal, Optional, Protocol, TypedDict

import nbformat
import requests
from unstructured.partition.md import partition_md
from unstructured.staging.base import convert_to_dict


# --- Type Definitions ---
class ElementMetadata(TypedDict):
    """Metadata for a single parsed element."""

    element_id: str
    text: str
    type: str
    page_number: Optional[int]
    coordinates: Optional[dict]
    metadata: Optional[dict]


class DocumentMetadata(TypedDict):
    """Metadata for the entire document."""

    file_path: str
    file_type: str
    structure: Literal["structured", "semi-structured", "unstructured", "empty"]
    document_type: str
    length: Literal["short", "medium", "long", "empty"]
    word_count: int
    page_count: Optional[int]
    elements: List[ElementMetadata]
    is_web_scraped: bool


class DocumentClassifier(Protocol):
    """Protocol for document type classifiers."""

    def classify(
        self, text: str, elements: List[ElementMetadata], file_type: str
    ) -> str:
        """Classify the document type (e.g., 'invoice', 'code', 'log')."""
        ...


# --- Helper Functions ---
def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def detect_file_type(file_path: str) -> str:
    """Detect file type from extension or URL."""
    if file_path.startswith(("http://", "https://")):
        return "html"
    suffix = Path(file_path).suffix.lower()
    type_map = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".html": "html",
        ".htm": "html",
        ".md": "md",
        ".mdx": "md",
        ".rst": "rst",
        ".py": "py",
        ".ipynb": "ipynb",
        ".log": "log",
        ".txt": "txt",
    }
    return type_map.get(suffix, "unknown")


def fetch_web_page(url: str) -> str:
    """Fetch HTML content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch URL {url}: {e}")


def parse_web_page(url: str) -> List[ElementMetadata]:
    """Parse a remote web page using auto partition with url= parameter."""
    logger.debug(f"parse_web_page | url={url}")
    elements = partition(url=url)
    return _convert_elements(elements)


def _parse_ipynb_elements(file_path: str) -> list:
    """Parse Jupyter notebook manually — not supported by auto partition."""
    with open(file_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    all_elements = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            sub_elements = partition_md(text=cell.source)
            all_elements.extend(sub_elements)
        elif cell.cell_type == "code":
            code_el = Text(text=cell.source)
            code_el.metadata.language = "python"
            code_el.metadata.cell_type = "code"
            all_elements.append(code_el)

    logger.debug(
        f"_parse_ipynb_elements | cells={len(notebook.cells)} | elements={len(all_elements)}"
    )
    return all_elements


def _convert_elements(raw_elements: list) -> List[ElementMetadata]:
    """Convert unstructured elements to typed dicts with traceability."""
    typed_elements = []
    for element in raw_elements:
        element_dict = convert_to_dict(element)
        typed_el = ElementMetadata(
            element_id=element_dict.get("element_id", str(hash(str(element)))),
            text=element_dict.get("text", ""),
            type=element_dict.get("type", "Unknown"),
            page_number=element_dict.get("metadata", {}).get("page_number"),
            coordinates=element_dict.get("metadata", {}).get("coordinates"),
            metadata=element_dict.get("metadata", {}),
        )
        typed_elements.append(typed_el)
    return typed_elements


def parse_document(file_path: str) -> List[ElementMetadata]:
    """
    Parse a document using auto partition for all supported types.
    Falls back to manual parsing only for .ipynb (not supported by auto).
    """
    file_type = detect_file_type(file_path)
    logger.debug(f"parse_document | path={file_path} | detected_type={file_type}")

    try:
        # .ipynb is NOT supported by auto partition — handle manually
        if file_type == "ipynb":
            logger.debug("parse_document | strategy=manual_ipynb (not in auto)")
            raw_elements = _parse_ipynb_elements(file_path)
        else:
            # ALL other types use unified auto partition
            # Auto detects file type via libmagic and routes internally
            logger.debug(f"parse_document | strategy=auto_partition(filename=)")
            raw_elements = partition(filename=file_path)

    except Exception as e:
        logger.error(
            f"parse_document | FAILED | type={file_type} | error={e}", exc_info=True
        )
        raise

    typed_elements = _convert_elements(raw_elements)

    logger.info(
        f"parse_document | SUCCESS | type={file_type} | "
        f"raw={len(raw_elements)} | typed={len(typed_elements)} | "
        f"categories={sorted(set(e['type'] for e in typed_elements))}"
    )
    return typed_elements


def classify_structure(
    elements: List[ElementMetadata], file_type: str
) -> Literal["structured", "semi-structured", "unstructured", "empty"]:
    """Classify document structure based on element types and file type."""
    if not elements:
        return "empty"

    if file_type in ("py", "ipynb"):
        return "structured"
    if file_type in ("md", "mdx", "rst"):
        return "semi-structured"
    if file_type == "log":
        return "unstructured"

    total_elements = len(elements)
    table_count = sum(1 for e in elements if e["type"] == "Table")
    list_count = sum(1 for e in elements if e["type"] == "ListItem")
    heading_count = sum(
        1 for e in elements if e["type"] in ("Title", "Header", "Subheader")
    )
    code_count = sum(1 for e in elements if e["type"] == "Code")

    table_ratio = table_count / total_elements
    heading_ratio = heading_count / total_elements
    code_ratio = code_count / total_elements

    if table_ratio > 0.3 or code_ratio > 0.5:
        return "structured"
    elif heading_ratio > 0.2 or (heading_count + list_count) / total_elements > 0.3:
        return "semi-structured"
    else:
        return "unstructured"


def classify_length(
    word_count: int, page_count: Optional[int] = None
) -> Literal["short", "medium", "long", "empty"]:
    """Classify document length."""
    if word_count == 0:
        return "empty"
    if page_count and page_count > 20:
        return "long"
    elif word_count > 5000:
        return "long"
    elif word_count > 500:
        return "medium"
    else:
        return "short"


# --- Classifiers ---
class KeywordDocumentClassifier:
    """Classify document type using keyword matching."""

    def __init__(self):
        self.keywords = {
            "invoice": ["invoice", "total", "amount due", "billing", "receipt", "$"],
            "research_paper": [
                "abstract",
                "methodology",
                "results",
                "references",
                "citation",
                "introduction",
            ],
            "email": ["subject:", "from:", "to:", "cc:", "best regards", "sincerely"],
            "contract": [
                "agreement",
                "party",
                "clause",
                "signatures",
                "terms and conditions",
            ],
            "medical_record": [
                "patient",
                "diagnosis",
                "treatment",
                "symptoms",
                "history",
            ],
            "code": ["def ", "class ", "import ", "function", "return ", "#!/usr/bin"],
            "log": ["error", "warning", "info", "debug", "exception", "traceback"],
            "web_page": ["<html", "<head", "<body", "DOCTYPE html"],
        }

    def classify(
        self, text: str, elements: List[ElementMetadata], file_type: str
    ) -> str:
        text_lower = text.lower()
        for doc_type, keywords in self.keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return doc_type
        if file_type == "py":
            return "code"
        elif file_type == "ipynb":
            return "code"
        elif file_type == "log":
            return "log"
        elif file_type == "html":
            return "web_page"
        elif file_type in ("md", "mdx", "rst"):
            return "markdown"
        else:
            return "unknown"


# --- Main Parser Class ---
class DocumentParser:
    """Parse and classify documents with type safety."""

    def __init__(self, classifier: Optional[DocumentClassifier] = None):
        self.classifier = classifier or KeywordDocumentClassifier()

    def parse_and_classify(self, file_path: str) -> DocumentMetadata:
        """Parse and classify with full error visibility."""
        logger.info(f"parse_and_classify | START | path={file_path}")

        try:
            elements = parse_document(file_path)
        except Exception as e:
            logger.error(
                f"parse_and_classify | PARSE_FAILED | path={file_path} | error={e}",
                exc_info=True,
            )
            return DocumentMetadata(
                file_path=file_path,
                file_type=detect_file_type(file_path),
                structure="empty",
                document_type="unknown",
                length="empty",
                word_count=0,
                page_count=None,
                elements=[],
                is_web_scraped=file_path.startswith(("http://", "https://")),
            )

        full_text = " ".join(e["text"] for e in elements)
        file_type = detect_file_type(file_path)
        structure = classify_structure(elements, file_type)
        document_type = self.classifier.classify(full_text, elements, file_type)
        word_count = count_words(full_text)
        page_count = next(
            (
                e["page_number"]
                for e in reversed(elements)
                if e.get("page_number") is not None
            ),
            None,
        )
        length = classify_length(word_count, page_count)

        result = DocumentMetadata(
            file_path=file_path,
            file_type=file_type,
            structure=structure,
            document_type=document_type,
            length=length,
            word_count=word_count,
            page_count=page_count,
            elements=elements,
            is_web_scraped=file_path.startswith(("http://", "https://")),
        )

        logger.info(
            f"parse_and_classify | DONE | type={document_type} | "
            f"structure={structure} | length={length} | "
            f"words={word_count} | elements={len(elements)}"
        )
        return result


# --- Example Usage ---
if __name__ == "__main__":
    parser = DocumentParser()

    test_files = [
        ("/Users/jethroestrada/Downloads/Resume Latest - Jethro Estrada.pdf", "PDF"),
        ("https://example.com", "Web URL"),
        (
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.html",
            "Local HTML",
        ),
        (
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.py",
            "Python",
        ),
        (
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.ipynb",
            "Jupyter",
        ),
        (
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.log",
            "Log",
        ),
        (
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.txt",
            "Empty TXT",
        ),
    ]

    logger.info("=" * 70)
    logger.info("BATCH DOCUMENT PARSING START")
    logger.info("=" * 70)

    results_summary = []
    for path, label in test_files:
        meta = parser.parse_and_classify(path)
        summary = (
            f"{label:12s} | type={meta['document_type']:15s} | "
            f"struct={meta['structure']:16s} | len={meta['length']:6s} | "
            f"elems={len(meta['elements'])}"
        )
        results_summary.append(summary)
        print(summary)

    logger.info("=" * 70)
    logger.info("BATCH COMPLETE")
    for line in results_summary:
        logger.info(line)
    logger.info("=" * 70)
