from typing import List, TypedDict
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.utils.embeddings import generate_embeddings
from jet.file.utils import load_file, save_file

import numpy as np
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class ContextItem(TypedDict):
    doc_idx: int
    tokens: int
    text: str

class SearchResult(TypedDict):
    rank: int
    doc_index: int
    score: float
    text: str

def search(
    query: str,
    documents: List[str],
    model: str | OLLAMA_MODEL_NAMES = "all-minilm:33m",
    top_k: int = None
) -> List[SearchResult]:
    """Search for documents most similar to the query.

    If top_k is None, return all results sorted by similarity.
    """
    if not documents:
        return []
    vectors = generate_embeddings([query] + documents, model, use_cache=True)
    query_vector = vectors[0]
    doc_vectors = vectors[1:]
    similarities = np.dot(doc_vectors, query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-10
    )
    sorted_indices = np.argsort(similarities)[::-1]
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]
    return [
        {
            "rank": i + 1,
            "doc_index": int(sorted_indices[i]),
            "score": float(similarities[sorted_indices[i]]),
            "text": documents[sorted_indices[i]],
        }
        for i in range(len(sorted_indices))
    ]


if __name__ == "__main__":
    all_contexts = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/all_contexts.json")
    urls = [
        "https://docs.tavily.com/documentation/api-reference/endpoint/crawl",
    ]
    model: OLLAMA_MODEL_NAMES = "embeddinggemma"

    # Search
    query = "How to change max depth?"
    texts = [doc["text"] for doc in all_contexts]
    search_results = search(query, texts, model)
    save_file({
        "query": query,
        "count": len(search_results),
        "results": search_results,
    }, f"{OUTPUT_DIR}/search_results.json")