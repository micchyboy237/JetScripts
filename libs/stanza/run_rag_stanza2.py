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

EXAMPLE_TEXT = (
    "OpenAI unveiled the GPT-5 model in October 2025, showcasing advanced reasoning "
    "and multilingual understanding capabilities. "
    "The model can process text, images, and structured data simultaneously. "
    "Analysts from Gartner believe this integration may redefine enterprise AI search. "
    "Meanwhile, universities such as MIT and ETH Zurich are testing Stanza-based parsing "
    "to improve context chunking for retrieval-augmented generation systems."
)


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
    example_parse_sentences()
    example_build_chunks()
    example_full_demo()
