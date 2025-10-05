import os
import shutil
import uuid
import numpy as np
from typing import List, Optional, TypedDict
from jet.file.utils import load_file, save_file
from jet.wordnet.text_chunker import chunk_texts_with_data
from jet.libs.llama_cpp.embeddings import LlamacppEmbedding
from jet.search.rag.base import preprocess_texts

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class SearchResult(TypedDict):
    id: str
    rank: int
    doc_index: int
    score: float
    tokens: int
    text: str

def search(
    query: str,
    documents: List[str],
    model: str = "embeddinggemma",
    top_k: int = None,
    ids: Optional[List[str]] = None,
    threshold: float = 0.0  # Minimum similarity threshold (default: 0.0)
) -> List[SearchResult]:
    """Search for documents most similar to the query.
    If top_k is None, return all results sorted by similarity.
    If ids is None, generate UUIDs for each document.
    Filters results based on threshold (only include if score >= threshold).
    """
    if not documents:
        return []
    client = LlamacppEmbedding(model=model)
    preprocessed_query = preprocess_texts(query)
    preprocessed_docs = preprocess_texts(documents)
    vectors = client.get_embeddings(preprocessed_query + preprocessed_docs, batch_size=16, show_progress=True)
    query_vector = vectors[0]
    doc_vectors = vectors[1:]
    similarities = np.dot(doc_vectors, query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-10
    )
    sorted_indices = np.argsort(similarities)[::-1]
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]
    
    # Apply threshold filter
    filtered_indices = [idx for idx in sorted_indices if similarities[idx] >= threshold]
    
    # Generate UUIDs if ids not provided, else use provided ids
    doc_ids = [str(uuid.uuid4()) for _ in documents] if ids is None else ids
    
    return [
        {
            "id": doc_ids[filtered_indices[i]],
            "rank": i + 1,
            "doc_index": int(filtered_indices[i]),
            "score": float(similarities[filtered_indices[i]]),
            "text": documents[filtered_indices[i]],
        }
        for i in range(len(filtered_indices))
    ]

def main(query: str, md_content: str, chunk_size: int, chunk_overlap: int, model: str = "embeddinggemma", threshold: float = 0.0):
    """Main function to process markdown content, chunk it, and perform search with optional threshold."""
    chunks = chunk_texts_with_data(md_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model)
    print(f"Number of chunks: {len(chunks)}")
    save_file(chunks, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/chunks.json")
    
    texts = [chunk["content"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    
    search_results = search(query, texts, model, ids=ids, threshold=threshold)
    
    # Add token count from original chunk
    for result in search_results:
        result["tokens"] = chunks[result["doc_index"]]["num_tokens"]
    
    return search_results

if __name__ == '__main__':
    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/https_docs_tavily_com_documentation_api_reference_endpoint_crawl/markdown_no_links.md"
    md_content: str = load_file(md_file)
    save_file(md_content, f"{OUTPUT_DIR}/doc.md")

    query = "How to change max depth?"
    model = "embeddinggemma"

    # Test various chunk sizes, overlaps, and thresholds
    chunk_sizes = [128, 64, 32, 16]
    chunk_overlaps = [64, 32, 16, 0]
    thresholds = [0.0]  # Test with different thresholds

    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            for threshold in thresholds:
                print(f"\n--- Searching with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, threshold={threshold} ---")
                search_results = main(query, md_content, chunk_size, chunk_overlap, model, threshold=threshold)
                save_file({
                    "model": model,
                    "query": query,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "threshold": threshold,
                    "count": len(search_results),
                    "results": search_results,
                }, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/threshold_{threshold}/search_results.json")
