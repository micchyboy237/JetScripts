"""Generic document parser using only unstructured auto partition."""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Set

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger("doc_parser")
logger.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s | %(message)s")
_stream = logging.StreamHandler(sys.stdout)
_stream.setFormatter(_formatter)
_file = logging.FileHandler(
    "/Users/jethroestrada/Desktop/External_Projects/"
    "Jet_Projects/JetScripts/test/__sample.log",
    mode="w",
    encoding="utf-8",
)
_file.setFormatter(_formatter)
logger.addHandler(_stream)
logger.addHandler(_file)

for _noisy in ("pdfminer", "urllib3", "PIL", "fontTools", "opencv"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

try:
    from unstructured.partition.auto import partition

    logger.info("✅ unstructured.partition.auto imported successfully")
except ImportError as e:
    logger.critical(f"❌ Missing dependency: {e}")
    logger.critical("   Fix: pip install 'unstructured[all-docs]'")
    sys.exit(1)

# High-value element types for RAG context (based on latest unstructured docs)
RAG_CONTEXT_TYPES: Set[str] = {
    "NarrativeText",
    "ListItem",
    "Title",
    "Header",
    "Table",
    "FigureCaption",
    "CodeSnippet",
    "Formula",
}


def _parse_notebook(path: str) -> List[Dict[str, Any]]:
    """
    Parse .ipynb files natively since unstructured's auto partition
    does not reliably produce semantic elements for notebooks.
    Converts markdown/code cells into unstructured-compatible element dicts.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        logger.error(f"_parse_notebook | FAILED | path={path} | error={e}")
        return []

    elements: List[Dict[str, Any]] = []
    cells = nb.get("cells", [])
    kernel_lang = nb.get("metadata", {}).get("language_info", {}).get("name", "")

    for cell in cells:
        cell_type = cell.get("cell_type", "")
        source_lines = cell.get("source", [])
        text = "".join(source_lines).strip()
        if not text:
            continue

        if cell_type == "markdown":
            # Split markdown into titles and narrative paragraphs
            for line in text.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    elements.append(
                        {
                            "type": "Title",
                            "text": stripped.lstrip("#").strip(),
                            "metadata": {},
                        }
                    )
                elif stripped.startswith("- ") or stripped.startswith("* "):
                    elements.append(
                        {
                            "type": "ListItem",
                            "text": stripped[2:].strip(),
                            "metadata": {},
                        }
                    )
                else:
                    elements.append(
                        {"type": "NarrativeText", "text": stripped, "metadata": {}}
                    )
        elif cell_type == "code":
            elements.append(
                {
                    "type": "CodeSnippet",
                    "text": text,
                    "metadata": {"language": kernel_lang},
                }
            )

    logger.info(f"_parse_notebook | parsed {len(elements)} elements from {path}")
    return elements


def extract_rag_context(elements: List[Dict[str, Any]]) -> str:
    """
    Extract clean RAG-ready text from parsed unstructured elements.

    Filters to high-value semantic element types, strips boilerplate,
    and joins into a single context string suitable for embedding or LLM input.

    Args:
        elements: List of element dicts (output of parse_document()["elements"])

    Returns:
        Cleaned, concatenated text containing only RAG-relevant content.
    """
    if not elements:
        return ""

    rag_parts: List[str] = []
    for elem in elements:
        elem_type = elem.get("type", "")
        text = elem.get("text", "").strip()
        if elem_type in RAG_CONTEXT_TYPES and text:
            if elem_type == "Table":
                rag_parts.append(f"[TABLE]\n{text}\n[/TABLE]")
            elif elem_type == "CodeSnippet":
                lang = elem.get("metadata", {}).get("language", "")
                header = f"[CODE{f' ({lang})' if lang else ''}]"
                rag_parts.append(f"{header}\n{text}\n[/CODE]")
            elif elem_type == "Formula":
                rag_parts.append(f"[FORMULA]{text}[/FORMULA]")
            else:
                rag_parts.append(text)

    context = "\n\n".join(rag_parts)
    logger.info(
        f"extract_rag_context | kept={len(rag_parts)} elements | "
        f"chars={len(context)} | types_used="
        f"{sorted({e.get('type') for e in elements if e.get('type') in RAG_CONTEXT_TYPES})}"
    )
    return context


def parse_document(path: str) -> Dict[str, Any]:
    """
    Parse any document using auto partition. Returns a standardized result dict.
    Handles local files, URLs, notebooks, and unsupported types gracefully.
    """
    logger.info(f"parse_document | START | path={path}")
    try:
        # ✅ Native notebook parsing — unstructured auto partition does not
        # reliably produce semantic elements for .ipynb files
        if path.endswith(".ipynb"):
            element_dicts = _parse_notebook(path)
            categories = sorted({e.get("type", "Unknown") for e in element_dicts})
            full_text = " ".join(e.get("text", "") for e in element_dicts)
            word_count = len(full_text.split())
            rag_context = extract_rag_context(element_dicts)
            result = {
                "path": path,
                "element_count": len(element_dicts),
                "categories": categories,
                "word_count": word_count,
                "page_count": None,
                "elements": element_dicts,
                "rag_context": rag_context,
                "status": "success",
            }
        else:
            if path.startswith(("http://", "https://")):
                elements = partition(url=path)
            else:
                elements = partition(filename=path)

            categories = [getattr(e, "category", "Unknown") for e in elements]
            full_text = " ".join(str(e) for e in elements)
            word_count = len(full_text.split())
            page_numbers = [
                getattr(e.metadata, "page_number", None)
                for e in elements
                if hasattr(e, "metadata")
            ]
            page_count = max((p for p in page_numbers if p is not None), default=None)

            element_dicts = [e.to_dict() for e in elements]
            rag_context = extract_rag_context(element_dicts)

            result = {
                "path": path,
                "element_count": len(elements),
                "categories": sorted(set(categories)),
                "word_count": word_count,
                "page_count": page_count,
                "elements": element_dicts,
                "rag_context": rag_context,
                "status": "success",
            }

        logger.info(
            f"parse_document | DONE | elements={result['element_count']} | "
            f"words={result['word_count']} | pages={result['page_count']} | "
            f"categories={result['categories']}"
        )
        return result
    except Exception as e:
        logger.error(
            f"parse_document | FAILED | path={path} | error={e}", exc_info=True
        )
        return {
            "path": path,
            "element_count": 0,
            "categories": [],
            "word_count": 0,
            "page_count": None,
            "elements": [],
            "rag_context": "",
            "status": f"error: {e}",
        }


if __name__ == "__main__":
    test_inputs = [
        "/Users/jethroestrada/Downloads/Resume Latest - Jethro Estrada.pdf",
        "https://example.com",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.html",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.py",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.ipynb",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.log",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.txt",
    ]

    logger.info("=" * 70)
    logger.info("BATCH START")
    logger.info("=" * 70)

    results = []
    for path in test_inputs:
        result = parse_document(path)
        label = path.split("/")[-1] if "/" in path else path
        status_icon = "✅" if result["status"] == "success" else "❌"
        rag_chars = len(result.get("rag_context", ""))
        summary = (
            f"{label:20s} | {status_icon} elems={result['element_count']:4d} | "
            f"words={result['word_count']:5d} | rag_chars={rag_chars:5d} | "
            f"cats={result['categories']}"
        )
        results.append(summary)
        print(summary)

    succeeded = sum(1 for r in results if "✅" in r)
    logger.info("=" * 70)
    logger.info(f"BATCH COMPLETE | {succeeded}/{len(results)} succeeded")
    for line in results:
        logger.info(line)
    logger.info("=" * 70)
