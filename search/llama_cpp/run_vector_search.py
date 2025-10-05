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
    ids: Optional[List[str]] = None
) -> List[SearchResult]:
    """Search for documents most similar to the query.
    If top_k is None, return all results sorted by similarity.
    If ids is None, generate UUIDs for each document.
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
    # Generate UUIDs if ids not provided, else use provided ids
    doc_ids = [str(uuid.uuid4()) for _ in documents] if ids is None else ids
    return [
        {
            "id": doc_ids[sorted_indices[i]],
            "rank": i + 1,
            "doc_index": int(sorted_indices[i]),
            "score": float(similarities[sorted_indices[i]]),
            "text": documents[sorted_indices[i]],
        }
        for i in range(len(sorted_indices))
    ]

def main(query: str, md_content: str, chunk_size: int, chunk_overlap: int, model: str = "embeddinggemma"):
    chunks = chunk_texts_with_data(md_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model)
    print(f"Number of chunks: {len(chunks)}")
    save_file(chunks, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/chunks.json")
    texts = [chunk["content"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    search_results = search(query, texts, model, ids=ids)
    for result in search_results:
        result["tokens"] = chunks[result["doc_index"]]["num_tokens"]
    return search_results

if __name__ == '__main__':
    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/https_docs_tavily_com_documentation_api_reference_endpoint_crawl/markdown_no_links.md"
    md_content: str = load_file(md_file)
    save_file(md_content, f"{OUTPUT_DIR}/doc.md")

    query = "How to change max depth?"
    model = "embeddinggemma"

    chunk_size = 128
    chunk_overlap = 32
    search_results = main(query, md_content, chunk_size, chunk_overlap, model)
    save_file({
        "model": model,
        "query": query,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "count": len(search_results),
        "results": search_results,
    }, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/search_results.json")
    
    chunk_size = 64
    chunk_overlap = 32
    search_results = main(query, md_content, chunk_size, chunk_overlap, model)
    save_file({
        "model": model,
        "query": query,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "count": len(search_results),
        "results": search_results,
    }, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/search_results.json")

    chunk_size = 64
    chunk_overlap = 16
    search_results = main(query, md_content, chunk_size, chunk_overlap, model)
    save_file({
        "model": model,
        "query": query,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "count": len(search_results),
        "results": search_results,
    }, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/search_results.json")
    
    chunk_size = 32
    chunk_overlap = 16
    search_results = main(query, md_content, chunk_size, chunk_overlap, model)
    save_file({
        "model": model,
        "query": query,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "count": len(search_results),
        "results": search_results,
    }, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/search_results.json")

    chunk_size = 32
    chunk_overlap = 0
    search_results = main(query, md_content, chunk_size, chunk_overlap, model)
    save_file({
        "model": model,
        "query": query,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "count": len(search_results),
        "results": search_results,
    }, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/search_results.json")

    chunk_size = 16
    chunk_overlap = 0
    search_results = main(query, md_content, chunk_size, chunk_overlap, model)
    save_file({
        "model": model,
        "query": query,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "count": len(search_results),
        "results": search_results,
    }, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/search_results.json")