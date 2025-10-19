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

from jet.logger import logger
from jet.libs.stanza.rag_stanza import (
    build_stanza_pipeline,
    parse_sentences,
    build_context_chunks,
    run_rag_stanza_demo,
)
from jet.libs.bertopic.examples.mock import load_sample_data_with_info
from jet.file.utils import save_file
from tqdm import tqdm
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

EXAMPLE_DATA = load_sample_data_with_info(model="embeddinggemma", chunk_size=512, chunk_overlap=64)
EXAMPLE_TEXT = EXAMPLE_DATA[5]["content"]


def example_build_pipeline():
    """Example: build the Stanza pipeline."""
    print("=== Example: Building Stanza pipeline ===")
    nlp = build_stanza_pipeline()
    print("Pipeline successfully created.")
    return nlp


def example_parse_sentences():
    """Example: parse text into structured sentence data."""
    print("\n=== Example: Parsing sentences ===")
    nlp = build_stanza_pipeline()
    parsed = parse_sentences(EXAMPLE_TEXT, nlp)
    print(f"Parsed {len(parsed)} sentences.")
    for i, s in enumerate(parsed[:3]):
        print(f"\n--- Sentence {i+1} ---")
        print(f"Text: {s['text']}")
        print(f"Tokens: {s['tokens']}")
        print(f"POS: {s['pos']}")
        print(f"Lemmas: {s['lemmas']}")
        print(f"Entities: {s['entities']}")
        print(f"Deps: {s['deps']}")
    return parsed


def example_build_chunks():
    """Example: build RAG context chunks from parsed sentences."""
    print("\n=== Example: Building context chunks ===")
    nlp = build_stanza_pipeline()
    parsed = parse_sentences(EXAMPLE_TEXT, nlp)
    chunks = build_context_chunks(parsed, max_tokens=50)
    print(f"Generated {len(chunks)} chunks.")
    for i, c in enumerate(chunks, 1):
        print(f"\n>>> Chunk {i}")
        print(f"Sentence indices: {c['sent_indices']}")
        print(f"Token count: {c['tokens']}")
        print(f"Salience: {c['salience']}")
        print(f"Entities: {', '.join(c['entities']) if c['entities'] else 'None'}")
        print(f"Text preview: {c['text'][:150]}...")
    return chunks


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
            "content": d["content"],
            "rag": results
        }
        yield final_result


if __name__ == "__main__":
    # Sequentially run all examples
    example_build_pipeline()

    token_counts = [chunk["num_tokens"] for chunk in EXAMPLE_DATA]
    save_file({
        "tokens": {
            "min": min(token_counts),
            "max": max(token_counts),
            "ave": sum(token_counts) // len(token_counts),
        }
    }, f"{OUTPUT_DIR}/chunks_info.json")
    save_file(EXAMPLE_DATA, f"{OUTPUT_DIR}/chunks.json")

    parsed_sentences_results = example_parse_sentences()
    save_file(parsed_sentences_results, f"{OUTPUT_DIR}/parsed_sentences_results.json")

    chunks_results = example_build_chunks()
    save_file(chunks_results, f"{OUTPUT_DIR}/chunks_results.json")

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
