import os
import shutil
import numpy as np
from typing import List, TypedDict
from jet.libs.llama_cpp.embeddings import LlamacppEmbedding
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class SearchResult(TypedDict):
    rank: int
    doc_index: int
    score: float
    text: str
    
def search(
    query: str,
    documents: List[str],
    model: str = "embeddinggemma-300M-Q8_0.gguf",
    top_k: int = None
) -> List[SearchResult]:
    """Search for documents most similar to the query.

    If top_k is None, return all results sorted by similarity.
    """
    if not documents:
        return []
    client = LlamacppEmbedding(model=model)
    vectors = client.get_embeddings([query] + documents, show_progress=True)
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

def main():
    """Example usage of EmbeddingClient."""
    model = "embeddinggemma-300M-Q8_0.gguf"
    
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
    ]
    
    search_results = search(query, documents, model)
    save_file({
        "query": query,
        "count": len(search_results),
        "results": search_results,
    }, f"{OUTPUT_DIR}/search_results.json")

if __name__ == "__main__":
    main()