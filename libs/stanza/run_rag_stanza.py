# jet_python_modules/jet/libs/stanza/examples/examples_rag_stanza.py
"""
examples_rag_stanza.py
Usage examples for `rag_stanza.py`.

Demonstrates how to:
  1. Build a Stanza pipeline.
  2. Parse sentences with syntax and entity metadata.
  3. Chunk parsed sentences for RAG.
  4. Run the full integrated demo.

Run:
    python examples_rag_stanza.py
"""

from contextlib import contextmanager
import time
from typing import Any, Dict, List
from jet.logger import logger
from jet.libs.stanza.rag_stanza import (
    build_stanza_pipeline,
    parse_sentences,
    build_context_chunks,
)
from jet.libs.bertopic.examples.mock import load_sample_data_with_info, ChunkResultWithMeta
from jet.file.utils import save_file
from tqdm import tqdm
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Sample web scraped data with headers separation chunking
EXAMPLE_DATA: List[ChunkResultWithMeta] = load_sample_data_with_info(model="embeddinggemma", chunk_size=512, chunk_overlap=64)

@contextmanager
def timer(description: str) -> None:
    """Context manager to track and print execution time."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.2f} seconds")

def run_rag_stanza_demo(text: str) -> Dict[str, Any]:
    """
    Run the full Stanza-based RAG preprocessing pipeline with progress tracking.
    Returns a dict with sentence-level and chunk-level information.
    """
    with timer("=== Building Stanza pipeline"):
        nlp = build_stanza_pipeline()
    
    with timer("=== Parsing sentences"):
        parsed_sentences = parse_sentences(text, nlp)
        print(f"Total sentences parsed: {len(parsed_sentences)}")
    
    with timer("=== Creating context chunks for RAG"):
        chunks = build_context_chunks(parsed_sentences, max_tokens=80)
        print(f"Generated {len(chunks)} chunks.\n")
    
    return {"parsed_sentences": parsed_sentences, "chunks": chunks}

def example_full_demo():
    """Example: run full integrated demo."""
    logger.debug("\n=== Example: Running full RAG Stanza demo ===")
    
    for i, d in enumerate(tqdm(EXAMPLE_DATA, desc="Processing RAG")):
        logger.info(f">>> Chunk {i + 1}")
        results = run_rag_stanza_demo(d["content"])
        logger.teal(f"\nParsed {len(results['parsed_sentences'])} sentences.")
        logger.teal(f"Built {len(results['chunks'])} chunks.")
        final_result = {
            "chunk_id": d["id"],
            "chunk_index": d["chunk_index"],
            "doc_id": d["doc_id"],
            "doc_index": d["doc_index"],
            "rag": results["chunks"],
            "parent_header": d["meta"]["parent_header"],
            "header": d["meta"]["header"],
            "content": d["content"],
            "sentences": results["parsed_sentences"],
        }
        yield final_result

if __name__ == "__main__":
    full_demo_stream = example_full_demo()
    all_results = []
    for result in full_demo_stream:
        all_results.append(result)
        out_path = f"{OUTPUT_DIR}/chunks/doc{result["doc_index"]}_chunk{result["chunk_index"]}.json"
        save_file(result, out_path)
    save_file({
        "count": len(all_results),
        "results": all_results
    }, f"{OUTPUT_DIR}/all_results.json")
