"""
examples_rag_stanza.py
======================

Usage examples for the Stanza-based RAG preprocessing utilities.

Demonstrates:
  - Pipeline caching and reuse
  - Sentence parsing and token-level features
  - Entity extraction and lemma normalization
  - Context chunk building for RAG
  - Query-guided context expansion
"""

from textwrap import shorten
from typing import List, Dict, TypedDict

from jet.file.utils import save_file
import stanza

from jet.libs.bertopic.examples.mock import load_sample_data_with_info
from jet.libs.stanza.rag_stanza import (
    StanzaPipelineCache,
    parse_sentences,
)

import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Sample web scraped data
EXAMPLE_DATA: List[str] = load_sample_data_with_info(model="embeddinggemma", chunk_size=512, truncate=True)
EXAMPLE_TEXT = EXAMPLE_DATA[4]["content"]


def build_cached_pipeline(lang: str = "en") -> stanza.Pipeline:
    """Create or reuse a cached Stanza pipeline."""
    cache = StanzaPipelineCache()
    pipeline = cache.get_pipeline(
        lang=lang,
        processors="tokenize,pos,lemma,depparse,ner",
        use_gpu=False
    )
    return pipeline


def example_build_rag_context(sentences: List[Dict], query: str = None) -> List[Dict]:
    """
    Example: Build retrieval-augmented context chunks.

    If a query is provided, prioritize sentences related to query entities or lemmas.
    """
    print("\n--- Building RAG Contexts ---")

    if query:
        query_terms = set(query.lower().split())
        filtered = [
            s for s in sentences
            if any(t["lemma"].lower() in query_terms for t in s["tokens"] if t["lemma"])
        ]
        print(f"Filtered {len(filtered)} relevant sentences based on query: '{query}'")
        contexts = filtered or sentences
    else:
        contexts = sentences

    # Format for RAG — small chunk dictionary
    rag_chunks = []
    for s in contexts:
        chunk_text = s["text"].strip()
        if len(chunk_text) < 30:
            continue
        rag_chunks.append({
            "id": s["sentence_id"],
            "text": chunk_text,
            "entities": [e["text"] for e in s.get("entities", [])],
            "token_count": s["token_count"],
        })

    for c in rag_chunks[:3]:  # preview only first 3 chunks
        print(f"[Chunk {c['id']}] {shorten(c['text'], width=120)}")
    return rag_chunks


from sklearn.metrics.pairwise import cosine_similarity


class SearchResult(TypedDict):
    rank: int
    doc_index: int
    score: float
    text: str


# Try using OllamaEmbeddings if available, otherwise fallback
try:
    from jet.llm.ollama.base_langchain import OllamaEmbeddings

    def get_embedding_model():
        print("\n[Embedding Model] Using OllamaEmbeddings backend...")
        return OllamaEmbeddings(model="nomic-embed-text")

except ImportError:
    from sentence_transformers import SentenceTransformer

    def get_embedding_model():
        print("\n[Embedding Model] Using SentenceTransformers fallback...")
        return SentenceTransformer("all-MiniLM-L6-v2")


def compute_embeddings(texts: List[str], model):
    """Compute embeddings for list of texts using chosen embedding model."""
    print(f"\nComputing embeddings for {len(texts)} texts...")
    if hasattr(model, "embed_documents"):  # Ollama-compatible
        return model.embed_documents(texts)
    elif hasattr(model, "encode"):  # SentenceTransformers
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    else:
        raise TypeError("Unsupported embedding model interface.")


def retrieve_similar_chunks(chunks: List[Dict], query: str, model, top_k: int = 3) -> List[SearchResult]:
    """Retrieve top-K chunks most similar to a query and return them as List[SearchResult]."""
    print("\n--- Running Similarity Search ---")
    texts = [c["text"] for c in chunks]

    query_vec = (
        model.embed_query(query)
        if hasattr(model, "embed_query")
        else model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    )
    chunk_vecs = compute_embeddings(texts, model)

    # Compute cosine similarity
    scores = cosine_similarity([query_vec], chunk_vecs)[0]
    # List of (idx, score) for each chunk
    indices_scores = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    top = indices_scores[:top_k]
    search_results: List[SearchResult] = []
    print(f"\nTop-{top_k} retrieved chunks for query: '{query}'")
    for rank_i, (doc_index, score) in enumerate(top, start=1):
        chunk = chunks[doc_index]
        print(f"({rank_i}) score={score:.3f} | {shorten(chunk['text'], width=100)}")
        search_results.append({
            "rank": rank_i,
            "doc_index": doc_index,
            "score": float(score),
            "text": chunk["text"]
        })
    return search_results


def example_end_to_end_rag_flow(query: str, text: str) -> List[SearchResult]:
    """Full example: parse → extract → chunk → embed → retrieve."""
    print("\n==================== RAG Flow Example ====================")
    pipeline = build_cached_pipeline()
    sentences = parse_sentences(text, pipeline)
    chunks = example_build_rag_context(sentences)

    # Step 1: Initialize embedding model
    model = get_embedding_model()

    # Step 2: Embed and retrieve
    results = retrieve_similar_chunks(chunks, query, model, top_k=5)

    return results

    
if __name__ == "__main__":
    save_file(EXAMPLE_DATA, f"{OUTPUT_DIR}/chunks.json")
    save_file(EXAMPLE_TEXT, f"{OUTPUT_DIR}/sample.txt")

    query = "Top isekai anime 2025"

    # Add full RAG flow demo
    search_results = example_end_to_end_rag_flow(query, EXAMPLE_TEXT)
    save_file(search_results, f"{OUTPUT_DIR}/search_results.json")
