from typing import Optional
from bs4 import BeautifulSoup
from jet.logger import logger
from jet.logger.timer import time_it
from jet.utils.commands import copy_to_clipboard
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class BertSearch:
    def __init__(self, model_name="paraphrase-MiniLM-L12-v2"):
        """Initialize the BERT model and FAISS index."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.doc_texts = []

    def _extract_text(self, html_content):
        """Extracts text from raw HTML content using BeautifulSoup."""
        soup = BeautifulSoup(html_content, "html.parser")
        return " ".join(p.text.strip() for p in soup.find_all("p") if p.text.strip())

    @time_it
    def build_index(self, html_docs, batch_size=32):
        """Processes and indexes a list of HTML documents with batch encoding."""
        self.doc_texts = [self._extract_text(doc) for doc in html_docs]

        # Generate embeddings with batch processing
        embeddings = self.model.encode(
            self.doc_texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

        # Create FAISS index using Inner Product (for cosine similarity)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

    @time_it
    def search(self, query: str, top_k: Optional[int] = None):
        """Searches the indexed documents for the best matches to the query."""
        if self.index is None:
            raise ValueError("Index is empty! Call build_index() first.")

        top_k = min(top_k or len(self.doc_texts), len(self.doc_texts))

        # Encode query using normalized embeddings
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True)

        # Perform search
        scores, indices = self.index.search(query_embedding, top_k)

        # Return top-k filtered results
        results = [{"text": self.doc_texts[idx], "score": round(float(score), 4)}
                   for idx, score in zip(indices[0], scores[0]) if idx < len(self.doc_texts)]

        return results


if __name__ == "__main__":
    # Sample HTML docs
    html_docs = [
        "<html><body><p>AI is transforming the world with deep learning.</p></body></html>",
        "<html><body><p>Quantum computing is the future of high-performance computing.</p></body></html>",
        "<html><body><p>Neural networks are a crucial part of artificial intelligence.</p></body></html>"
    ]

    # Initialize search system
    search_engine = BertSearch()

    # Index the documents
    search_engine.build_index(html_docs)

    # Perform a search
    query = "Tell me about AI and deep learning"
    results = search_engine.search(query, top_k=5)

    # Print results
    for i, res in enumerate(results):
        print(f"Rank {i+1}: {res['text']} (Score: {res['score']})")

    copy_to_clipboard({
        "count": len(results),
        "data": results[:50]
    })

    for idx, result in enumerate(results[:10]):
        logger.log(f"{idx + 1}:", result["text"]
                   [:30], colors=["WHITE", "DEBUG"])
        logger.success(f"{result['score']:.2f}")
