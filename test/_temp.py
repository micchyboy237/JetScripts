import os
import shutil
import numpy as np
from typing import List, TypedDict
from jet.file.utils import load_file, save_file
from jet.wordnet.text_chunker import chunk_texts_with_data
from jet.libs.llama_cpp.embeddings import LlamacppEmbedding

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class SearchResult(TypedDict):
    rank: int
    doc_index: int
    score: float
    tokens: int
    text: str

def search(
    query: str,
    documents: List[str],
    model: str = "nomic-embed-text-v2-moe",
    top_k: int = None
) -> List[SearchResult]:
    """Search for documents most similar to the query.

    If top_k is None, return all results sorted by similarity.
    """
    if not documents:
        return []
    client = LlamacppEmbedding(model=model)
    vectors = client.get_embeddings([query] + documents, batch_size=32, show_progress=True)
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

if __name__ == '__main__':
    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/https_docs_tavily_com_documentation_api_reference_endpoint_crawl/markdown.md"
    md_content: str = load_file(md_file)

    model = "nomic-embed-text-v2-moe"
    query = "How to change max depth?"

    chunks = chunk_texts_with_data(md_content, chunk_size=128, chunk_overlap=32, model=model)
    print(f"Number of chunks: {len(chunks)}")
    save_file(chunks, f"{OUTPUT_DIR}/chunks.json")

    texts = [chunk["content"] for chunk in chunks]
    search_results = search(query, texts, model)
    
    for result in search_results:
        result["tokens"] = chunks[result["doc_index"]]["num_tokens"]

    save_file({
        "model": model,
        "query": query,
        "count": len(search_results),
        "results": search_results,
    }, f"{OUTPUT_DIR}/search_results.json")
