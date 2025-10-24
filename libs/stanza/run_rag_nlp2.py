"""
examples_rag_pipeline.py
----------------------------------
Usage examples for the RAG pipeline using Stanza NLP + SentenceTransformers.

This file demonstrates:
1. Markdown preprocessing and sentence splitting
2. Chunking long documents
3. Embedding and retrieval (with and without MMR)
4. Optional NER extraction via Stanza

Run:
    python examples_rag_pipeline.py
"""

from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.libs.bertopic.examples.mock import load_sample_md_doc
from jet.libs.stanza.rag_nlp2 import (
    init_stanza_pipeline,
    preprocess_markdown_to_sentences,
    chunk_sentences,
    retrieve,
)
from sentence_transformers import SentenceTransformer


def example_basic_retrieval() -> None:
    """
    Example 1: Basic semantic retrieval without MMR diversification.
    """
    print("\n=== Example 1: Basic Retrieval ===")

    markdown_doc = load_sample_md_doc()
    query = "Top isekai anime 2025"

    # When we preprocess the markdown text
    sentences = preprocess_markdown_to_sentences(markdown_doc)
    print(f"Extracted {len(sentences)} sentences")

    # And chunk the sentences into manageable sections
    chunks = chunk_sentences(sentences, max_chars=250)
    print(f"Created {len(chunks)} chunks")

    # Load an embedding model (generic and fast)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Then retrieve top relevant chunks
    results = retrieve(query, chunks, embedding_model=model, mmr=False, mmr_top_k=10)

    # Display retrieved results
    for i, r in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Similarity Score: {r['score']['similarity']:.4f}")
        print(f"Chunk Text: {r['meta']['text'][:150]}...")


def example_mmr_retrieval() -> None:
    """
    Example 2: Retrieval with MMR (diversity-aware ranking).
    """
    print("\n=== Example 2: MMR Retrieval ===")

    markdown_doc = load_sample_md_doc()
    query = "Top isekai anime 2025"

    sentences = preprocess_markdown_to_sentences(markdown_doc)
    chunks = chunk_sentences(sentences, max_chars=200)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results = retrieve(query, chunks, embedding_model=model, mmr=True, mmr_top_k=3, mmr_lambda=0.6)

    for i, r in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Similarity: {r['score']['similarity']:.4f}")
        print(f"Diversity:  {r['score']['diversity']}")
        print(f"Chunk Text: {r['meta']['text'][:120]}...")


def example_with_stanza_ner() -> None:
    """
    Example 3: Retrieval with Named Entity Recognition (NER) using Stanza.
    """
    print("\n=== Example 3: Retrieval with NER (Stanza) ===")

    markdown_doc = load_sample_md_doc()
    query = "Top isekai anime 2025"

    # Initialize Stanza (English pipeline)
    stanza_pipe = init_stanza_pipeline("en")

    sentences = preprocess_markdown_to_sentences(markdown_doc, stanza_pipe=stanza_pipe)
    chunks = chunk_sentences(sentences, max_chars=200)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "Which companies are involved in AI development?"

    results = retrieve(query, chunks, embedding_model=model, mmr=False, stanza_pipe=stanza_pipe)

    for i, r in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Similarity: {r['score']['similarity']:.4f}")
        print(f"Chunk Text: {r['meta']['text'][:150]}...")
        if r["ner"]:
            print("Named Entities:", [f"{e['text']} ({e['type']})" for e in r["ner"]])
        else:
            print("No NER entities detected.")


def example_multi_document_retrieval() -> None:
    """
    Example 4: Query across multiple markdown documents.
    """
    print("\n=== Example 4: Multi-document Retrieval ===")

    markdown_doc = load_sample_md_doc()
    docs = derive_by_header_hierarchy(markdown_doc, ignore_links=True)
    query = "Top isekai anime 2025"

    # Preprocess and chunk each doc
    all_chunks = []
    for d in docs:
        text = f"{d["header"]}\n{d["content"]}"
        sents = preprocess_markdown_to_sentences(text)
        chunks = chunk_sentences(sents, max_chars=200)
        for c in chunks:
            c["doc_id"] = d["id"]
        all_chunks.extend(chunks)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "Explain how hybrid computing works"

    results = retrieve(query, all_chunks, embedding_model=model, mmr=True, mmr_top_k=3, mmr_lambda=0.65)

    for i, r in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Doc: {r['meta']['doc_id']}")
        print(f"Similarity: {r['score']['similarity']:.4f}")
        print(f"Diversity:  {r['score']['diversity']}")
        print(f"Chunk: {r['meta']['text'][:120]}...")


if __name__ == "__main__":
    # Demonstrate all examples
    example_basic_retrieval()
    example_mmr_retrieval()
    example_with_stanza_ner()
    example_multi_document_retrieval()
