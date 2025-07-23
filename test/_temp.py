from typing import List, Dict, Any
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer, util
import numpy as np


class SpellCorrectedSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize spell checker and sentence transformer model."""
        self.spell_checker = SpellChecker()
        self.model = SentenceTransformer(model_name)
        self.documents: List[Dict[str, Any]] = []
        self.corrected_documents: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None

    def correct_text(self, text: str) -> str:
        """Correct misspellings in a given text."""
        words = text.split()
        corrected_words = [self.spell_checker.correction(
            word) or word for word in words]
        return " ".join(corrected_words)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents, correct misspellings, and compute embeddings."""
        self.documents = documents
        self.corrected_documents = [
            {"id": doc["id"], "content": self.correct_text(doc["content"])}
            for doc in documents
        ]
        texts = [doc["content"] for doc in self.corrected_documents]
        self.embeddings = self.model.encode(texts, convert_to_tensor=True)

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on corrected documents."""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(cos_scores.cpu().numpy())[::-1][:limit]

        results = []
        for idx in top_indices:
            score = cos_scores[idx].item()
            # Return original document with misspellings
            doc = self.documents[idx]
            results.append(
                {"id": doc["id"], "content": doc["content"], "score": score})
        return results


def main():
    # Sample documents with misspellings
    documents = [
        {"id": 1, "content": "The quick brown foxx jumps over the lazy dog"},
        {"id": 2, "content": "A beautifull garden blooms with collorful flowers"},
        {"id": 3, "content": "Teh sun sets slowly behind the mountan"},
    ]
    keywords = ["beautiful garden", "quick fox", "sunset mountain"]

    # Initialize search engine
    search_engine = SpellCorrectedSearchEngine()
    search_engine.add_documents(documents)

    # Perform searches
    for keyword in keywords:
        results = search_engine.search(keyword)
        print(f"\nSearch query: {keyword}")
        for result in results:
            print(
                f"Document {result['id']}: {result['content']} (Score: {result['score']:.2f})")


if __name__ == "__main__":
    main()
