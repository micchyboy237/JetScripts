from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorSearch:
    """A vector search engine using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.documents: List[str] = []
        self.vectors: np.ndarray = None
        self.model = SentenceTransformer(model_name)

    def add_documents(self, documents: List[str]) -> None:
        """Add documents and their vector representations."""
        self.documents = documents
        self.vectors = self.model.encode(documents)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for documents most similar to the query."""
        if not self.documents:
            return []

        query_vector = self.model.encode([query])[0]
        similarities = np.dot(self.vectors, query_vector) / (
            np.linalg.norm(self.vectors, axis=1) *
            np.linalg.norm(query_vector) + 1e-10
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.documents[i], similarities[i]) for i in top_indices]


if __name__ == "__main__":
    # Real-world demonstration
    search_engine = VectorSearch()

    # Same sample documents
    sample_docs = [
        "Fresh organic apples from local farms",
        "Handpicked strawberries sweet and juicy",
        "Premium quality oranges rich in vitamin C",
        "Crisp lettuce perfect for salads",
        "Organic bananas ripe and ready to eat"
    ]

    search_engine.add_documents(sample_docs)

    # Same example queries
    queries = [
        "organic fruit",
        "sweet strawberries",
        "fresh salad ingredients"
    ]

    for query in queries:
        results = search_engine.search(query)
        print(f"\nQuery: {query}")
        print("Top matches:")
        for doc, score in results:
            print(f"- {doc} (score: {score:.3f})")
