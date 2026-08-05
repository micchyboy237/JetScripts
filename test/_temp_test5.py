"""Generic document parser with hierarchical RAG chunking for small-context LLMs."""

import json
import logging
import os
import re
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

# High-value element types for RAG context
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

# Atomic types that must never be split mid-element
ATOMIC_TYPES: Set[str] = {"Table", "CodeSnippet", "Formula"}

# Section boundary types
SECTION_TYPES: Set[str] = {"Title", "Header"}


# ---------------------------------------------------------------------------
# Token estimation (lightweight, no external deps required)
# Replace with tiktoken or llama-cpp tokenizer for production accuracy
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Estimate token count. ~4 chars/token for English; override for production."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Structure classification
# ---------------------------------------------------------------------------
def classify_structure(elements: List[Dict[str, Any]]) -> str:
    """Classify document structure to select chunking strategy."""
    if not elements:
        return "empty"
    types = [e.get("type", "") for e in elements]
    has_sections = any(t in SECTION_TYPES for t in types)
    has_atomic = any(t in ATOMIC_TYPES for t in types)
    narrative_count = sum(1 for t in types if t == "NarrativeText")
    total = len(types)

    if has_sections and total > 3:
        return "structured"
    elif has_atomic and not has_sections:
        return "atomic_flat"
    elif narrative_count == total and total > 0:
        return "flat_narrative"
    elif total <= 2:
        return "monolithic"
    else:
        return "mixed"


# ---------------------------------------------------------------------------
# Sentence splitting helper
# ---------------------------------------------------------------------------
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences. Returns single-item list if no boundaries found."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()] or [text.strip()]


# ---------------------------------------------------------------------------
# Core chunking engine
# ---------------------------------------------------------------------------
def _merge_up_to_budget(
    texts: List[str], max_tokens: int, overlap_tokens: int
) -> List[Dict[str, Any]]:
    """Merge sequential text fragments into chunks respecting token budget."""
    chunks: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_tokens = 0

    for text in texts:
        text_tokens = estimate_tokens(text)
        # If single fragment exceeds budget, force-split it
        if text_tokens > max_tokens:
            # Flush current buffer first
            if current_parts:
                chunks.append(
                    {
                        "text": "\n\n".join(current_parts),
                        "token_count": current_tokens,
                    }
                )
                current_parts = []
                current_tokens = 0
            # Hard-split oversized fragment at word boundaries
            words = text.split()
            sub_parts: List[str] = []
            sub_tokens = 0
            for word in words:
                w_tok = estimate_tokens(word)
                if sub_tokens + w_tok > max_tokens and sub_parts:
                    chunks.append(
                        {
                            "text": " ".join(sub_parts),
                            "token_count": sub_tokens,
                        }
                    )
                    # Overlap: carry tail of previous chunk
                    if overlap_tokens > 0:
                        overlap_text = " ".join(sub_parts[-3:])  # approx overlap
                        sub_parts = [overlap_text]
                        sub_tokens = estimate_tokens(overlap_text)
                    else:
                        sub_parts = []
                        sub_tokens = 0
                sub_parts.append(word)
                sub_tokens += w_tok
            if sub_parts:
                chunks.append(
                    {
                        "text": " ".join(sub_parts),
                        "token_count": sub_tokens,
                    }
                )
            continue

        # Normal case: fits within budget
        if current_tokens + text_tokens > max_tokens and current_parts:
            chunks.append(
                {
                    "text": "\n\n".join(current_parts),
                    "token_count": current_tokens,
                }
            )
            # Overlap: carry last fragment(s) into next chunk
            if overlap_tokens > 0:
                overlap_parts: List[str] = []
                overlap_tok = 0
                for part in reversed(current_parts):
                    pt = estimate_tokens(part)
                    if overlap_tok + pt > overlap_tokens:
                        break
                    overlap_parts.insert(0, part)
                    overlap_tok += pt
                current_parts = overlap_parts
                current_tokens = overlap_tok
            else:
                current_parts = []
                current_tokens = 0

        current_parts.append(text)
        current_tokens += text_tokens

    if current_parts:
        chunks.append(
            {
                "text": "\n\n".join(current_parts),
                "token_count": current_tokens,
            }
        )
    return chunks


def chunk_rag_context(
    elements: List[Dict[str, Any]],
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> List[Dict[str, Any]]:
    """
    Hierarchical, token-aware chunking of parsed document elements.

    Automatically selects strategy based on document structure:
      - structured: section → paragraph → sentence hierarchy
      - flat_narrative: synthetic paragraph grouping → sentence fallback
      - atomic_flat: each atomic element = own chunk; narrative grouped separately
      - monolithic: sentence splitting → merge up to budget
      - mixed: anchored sections + flat fallback for unanchored runs

    Args:
        elements: List of element dicts from extract_rag_context pipeline
        max_tokens: Target max tokens per chunk (256-400 for small-context LLMs)
        overlap_tokens: Overlap between consecutive chunks (0 for sparse retrieval)

    Returns:
        List of chunk dicts with 'text', 'token_count', and 'strategy' keys.
    """
    if not elements:
        return []

    # Filter to RAG-relevant types only
    rag_elements = [
        e
        for e in elements
        if e.get("type") in RAG_CONTEXT_TYPES and e.get("text", "").strip()
    ]
    if not rag_elements:
        return []

    structure = classify_structure(rag_elements)
    logger.info(
        f"chunk_rag_context | structure={structure} | elements={len(rag_elements)} | max_tokens={max_tokens}"
    )

    chunks: List[Dict[str, Any]] = []

    if structure == "structured":
        # Group by section boundaries
        sections: List[List[Dict]] = [[]]
        for elem in rag_elements:
            if elem.get("type") in SECTION_TYPES and sections[-1]:
                sections.append([])
            sections[-1].append(elem)

        for section_elems in sections:
            if not section_elems:
                continue
            # Atomic elements become standalone chunks
            narrative_parts: List[str] = []
            for elem in section_elems:
                etype = elem.get("type", "")
                text = elem.get("text", "").strip()
                if etype in ATOMIC_TYPES:
                    # Flush accumulated narrative first
                    if narrative_parts:
                        chunks.extend(
                            _merge_up_to_budget(
                                narrative_parts, max_tokens, overlap_tokens
                            )
                        )
                        narrative_parts = []
                    # Atomic = own chunk (never split)
                    tok = estimate_tokens(text)
                    chunks.append(
                        {"text": text, "token_count": tok, "strategy": "atomic"}
                    )
                else:
                    narrative_parts.append(text)
            # Flush remaining narrative in section
            if narrative_parts:
                chunks.extend(
                    _merge_up_to_budget(narrative_parts, max_tokens, overlap_tokens)
                )

    elif structure == "atomic_flat":
        narrative_parts: List[str] = []
        for elem in rag_elements:
            etype = elem.get("type", "")
            text = elem.get("text", "").strip()
            if etype in ATOMIC_TYPES:
                if narrative_parts:
                    chunks.extend(
                        _merge_up_to_budget(narrative_parts, max_tokens, overlap_tokens)
                    )
                    narrative_parts = []
                chunks.append(
                    {
                        "text": text,
                        "token_count": estimate_tokens(text),
                        "strategy": "atomic",
                    }
                )
            else:
                narrative_parts.append(text)
        if narrative_parts:
            chunks.extend(
                _merge_up_to_budget(narrative_parts, max_tokens, overlap_tokens)
            )

    elif structure == "flat_narrative":
        # Synthetic paragraph grouping: every 3-5 sequential elements
        GROUP_SIZE = 4
        groups: List[List[str]] = []
        texts = [e.get("text", "").strip() for e in rag_elements]
        for i in range(0, len(texts), GROUP_SIZE):
            groups.append(texts[i : i + GROUP_SIZE])
        for group in groups:
            merged = "\n\n".join(group)
            if estimate_tokens(merged) <= max_tokens:
                chunks.append(
                    {
                        "text": merged,
                        "token_count": estimate_tokens(merged),
                        "strategy": "synthetic_para",
                    }
                )
            else:
                # Split within group at sentence boundaries
                sentences: List[str] = []
                for t in group:
                    sentences.extend(split_sentences(t))
                chunks.extend(
                    _merge_up_to_budget(sentences, max_tokens, overlap_tokens)
                )

    elif structure == "monolithic":
        text = rag_elements[0].get("text", "").strip()
        sentences = split_sentences(text)
        if len(sentences) == 1 and estimate_tokens(text) <= max_tokens:
            chunks.append(
                {
                    "text": text,
                    "token_count": estimate_tokens(text),
                    "strategy": "monolithic",
                }
            )
        else:
            chunks.extend(_merge_up_to_budget(sentences, max_tokens, overlap_tokens))
            for c in chunks:
                c["strategy"] = "monolithic_split"

    else:  # mixed
        # Use available Titles as anchors; unanchored content gets flat treatment
        anchored_run: List[Dict] = []
        flat_run: List[str] = []
        for elem in rag_elements:
            if elem.get("type") in SECTION_TYPES:
                # Flush flat run
                if flat_run:
                    chunks.extend(
                        _merge_up_to_budget(flat_run, max_tokens, overlap_tokens)
                    )
                    flat_run = []
                anchored_run = [elem]
            elif anchored_run:
                anchored_run.append(elem)
            else:
                flat_run.append(elem.get("text", "").strip())

        # Process final anchored run as structured
        if anchored_run:
            narrative_parts: List[str] = []
            for elem in anchored_run:
                etype = elem.get("type", "")
                text = elem.get("text", "").strip()
                if etype in ATOMIC_TYPES:
                    if narrative_parts:
                        chunks.extend(
                            _merge_up_to_budget(
                                narrative_parts, max_tokens, overlap_tokens
                            )
                        )
                        narrative_parts = []
                    chunks.append(
                        {
                            "text": text,
                            "token_count": estimate_tokens(text),
                            "strategy": "atomic",
                        }
                    )
                else:
                    narrative_parts.append(text)
            if narrative_parts:
                chunks.extend(
                    _merge_up_to_budget(narrative_parts, max_tokens, overlap_tokens)
                )
        # Flush trailing flat run
        if flat_run:
            chunks.extend(_merge_up_to_budget(flat_run, max_tokens, overlap_tokens))

    # Tag all chunks with strategy if not already set
    for c in chunks:
        c.setdefault("strategy", structure)

    logger.info(
        f"chunk_rag_context | produced {len(chunks)} chunks | "
        f"total_tokens={sum(c['token_count'] for c in chunks)} | "
        f"strategies={sorted({c['strategy'] for c in chunks})}"
    )
    return chunks


# ---------------------------------------------------------------------------
# Notebook parser (unchanged)
# ---------------------------------------------------------------------------
def _parse_notebook(path: str) -> List[Dict[str, Any]]:
    """Parse .ipynb natively into unstructured-compatible element dicts."""
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


# ---------------------------------------------------------------------------
# RAG context extraction (unchanged)
# ---------------------------------------------------------------------------
def extract_rag_context(elements: List[Dict[str, Any]]) -> str:
    """Extract clean RAG-ready text from parsed elements."""
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


# ---------------------------------------------------------------------------
# Document parser (now includes chunking)
# ---------------------------------------------------------------------------
def parse_document(
    path: str,
    chunk_max_tokens: int = 400,
    chunk_overlap_tokens: int = 50,
) -> Dict[str, Any]:
    """
    Parse any document and produce RAG-ready chunks.
    Handles local files, URLs, notebooks, and unsupported types gracefully.
    """
    logger.info(f"parse_document | START | path={path}")
    try:
        if path.endswith(".ipynb"):
            element_dicts = _parse_notebook(path)
            categories = sorted({e.get("type", "Unknown") for e in element_dicts})
            full_text = " ".join(e.get("text", "") for e in element_dicts)
            word_count = len(full_text.split())
            page_count = None
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
        chunks = chunk_rag_context(
            element_dicts, chunk_max_tokens, chunk_overlap_tokens
        )

        result = {
            "path": path,
            "element_count": len(element_dicts),
            "categories": sorted(set(categories)),
            "word_count": word_count,
            "page_count": page_count,
            "elements": element_dicts,
            "rag_context": rag_context,
            "chunks": chunks,
            "status": "success",
        }
        logger.info(
            f"parse_document | DONE | elements={len(element_dicts)} | "
            f"words={word_count} | pages={page_count} | "
            f"chunks={len(chunks)} | categories={result['categories']}"
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
            "chunks": [],
            "status": f"error: {e}",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
        n_chunks = len(result.get("chunks", []))
        total_chunk_tokens = sum(c["token_count"] for c in result.get("chunks", []))
        summary = (
            f"{label:20s} | {status_icon} elems={result['element_count']:4d} | "
            f"chunks={n_chunks:3d} | tokens={total_chunk_tokens:5d} | "
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
