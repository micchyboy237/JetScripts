"""
examples_rag_stanza.py

Demonstrates how to use `rag_stanza` on long, realistic inputs to:
  1. Build a full Stanza pipeline.
  2. Parse text into syntactically and semantically rich sentence data.
  3. Chunk text for use in a Retrieval-Augmented Generation (RAG) system.
  4. Inspect metadata such as entities and salience scores.

Run:
    python examples_rag_stanza.py
"""

from jet.libs.stanza.rag_stanza_backup import build_stanza_pipeline, parse_sentences, chunk_sentences_for_rag, build_context_chunks

# --- Example 1: Long multi-paragraph input ---
LONG_TEXT = """
OpenAI announced the GPT-5 model on October 15, 2025, marking a major leap in multimodal reasoning and multilingual understanding.
The company claims that GPT-5 can handle text, image, and structured data simultaneously, offering developers a unified API for advanced RAG systems.
Industry analysts from Gartner and McKinsey noted that the integration of reasoning and retrieval may redefine enterprise search.

In a blog post, OpenAI also shared new benchmarks showing 40% higher factual accuracy when used with domain-specific retrieval pipelines.
For instance, a medical RAG system fine-tuned on PubMed articles showed improved diagnostic explanations with lower hallucination rates.

Meanwhile, universities like Stanford, MIT, and ETH Zurich are testing open-source alternatives built on Stanza and Hugging Face Transformers,
demonstrating that combining syntactic dependency parsing with retrieval improves contextual chunking and precision recall trade-offs.

Experts believe that syntax-aware chunking can drastically reduce embedding redundancy,
allowing context windows to fit more meaningful text and reducing retrieval noise for downstream large language models.
"""

def example_full_pipeline():
    """
    Run a full end-to-end demo using Stanza to build RAG-ready chunks.
    """

    print("=== Building Stanza pipeline (this may take a few seconds) ===")
    pipeline = build_stanza_pipeline(lang="en")

    print("\n=== Parsing sentences ===")
    sentences = parse_sentences(LONG_TEXT, pipeline)
    print(f"Total sentences parsed: {len(sentences)}")

    # Show a few representative sentence analyses
    for i, s in enumerate(sentences[:3]):
        print(f"\n--- Sentence {i+1} ---")
        print(f"Text: {s['text']}")
        print(f"Tokens: {s['tokens']}")
        print(f"POS: {s['pos']}")
        print(f"Lemmas: {s['lemmas'][:6]}...")
        if s['entities']:
            print(f"Entities: {[e['text'] + ':' + e['type'] for e in s['entities']]}")
        print(f"Dependency heads: {[d['deprel'] for d in s['deps']]}")

    print("\n=== Creating context chunks for RAG ===")
    chunks = chunk_sentences_for_rag(sentences, max_tokens=60, overlap=15)
    print(f"Generated {len(chunks)} chunks.\n")

    for i, c in enumerate(chunks):
        print(f"\n>>> Chunk {i+1}")
        print(f"Sentences: {c['sentence_indices']}")
        print(f"Token count (approx): {c['est_token_count']}")
        print(f"Salience score: {c['metadata']['salience']:.2f}")
        print(f"Text preview: {c['text'][:200]}...")
        if c['metadata'].get("entities"):
            ents = [e['text'] for e in c['metadata']['entities']]
            print(f"Entities in chunk: {', '.join(set(ents))}")

def example_quick_pipeline():
    """
    Simplified version using the high-level `build_context_chunks` function.
    """
    print("\n=== Using build_context_chunks() ===")
    chunks = build_context_chunks(LONG_TEXT, lang="en", max_tokens=80, overlap=20)
    print(f"Chunks built: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"\nChunk {i+1} summary:")
        print(f"- Tokens ≈ {c['est_token_count']}")
        print(f"- Sentences: {len(c['sentence_indices'])}")
        print(f"- Salience: {c['metadata']['salience']:.2f}")
        if c['metadata'].get("entities"):
            ents = [e['text'] for e in c['metadata']['entities']]
            print(f"- Entities: {', '.join(sorted(set(ents)))}")
        print(f"- Text (first 150 chars): {c['text'][:150]}...")

if __name__ == "__main__":
    example_full_pipeline()
    example_quick_pipeline()
