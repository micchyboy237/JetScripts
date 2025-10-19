# jet_python_modules/jet/libs/stanza/examples/examples_rag_stanza2.py
"""
examples_rag_stanza2.py
Usage examples for `rag_stanza2.py`.

Demonstrates how to:
  1. Build a Stanza pipeline.
  2. Parse sentences with syntax and entity metadata.
  3. Chunk parsed sentences for RAG.
  4. Run the full integrated demo.

Run:
    python examples_rag_stanza2.py
"""

from jet.libs.stanza.rag_stanza2 import (
    build_stanza_pipeline,
    parse_sentences,
    build_context_chunks,
    run_rag_stanza_demo,
)
from jet.libs.bertopic.examples.mock import load_sample_data
from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

EXAMPLE_TEXT = load_sample_data(model="embeddinggemma", chunk_size=512, truncate=True)[5]


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
    print("\n=== Example: Running full RAG Stanza demo ===")
    results = run_rag_stanza_demo(EXAMPLE_TEXT)
    print(f"\nParsed {len(results['parsed_sentences'])} sentences.")
    print(f"Built {len(results['chunks'])} chunks.")
    return results


if __name__ == "__main__":
    # Sequentially run all examples
    example_build_pipeline()

    parsed_sentences_results = example_parse_sentences()
    save_file(parsed_sentences_results, f"{OUTPUT_DIR}/parsed_sentences_results.json")

    chunks_results = example_build_chunks()
    save_file(chunks_results, f"{OUTPUT_DIR}/chunks_results.json")

    full_demo_results = example_full_demo()
    save_file(full_demo_results, f"{OUTPUT_DIR}/full_demo_results.json")
