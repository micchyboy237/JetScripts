import math
from collections import Counter
from typing import List, Dict


class BM25Reranker:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        """
        BM25 Reranker with term saturation and document length normalization.
        :param documents: List of text documents (corpus)
        :param k1: Term saturation parameter (default 1.5)
        :param b: Document length normalization parameter (default 0.75)
        """
        self.documents: List[str] = documents
        self.k1: float = k1
        self.b: float = b
        self.doc_lengths: List[int] = [len(doc.split()) for doc in documents]
        self.avg_doc_length: float = sum(self.doc_lengths) / len(documents)
        self.inverted_index: Dict[str, List[int]
                                  ] = self._build_inverted_index()
        self.idf: Dict[str, float] = self._compute_idf()

    def _build_inverted_index(self) -> Dict[str, List[int]]:
        index: Dict[str, List[int]] = {}
        for doc_id, doc in enumerate(self.documents):
            for term in set(doc.split()):  # Unique terms only for IDF calculation
                index.setdefault(term, []).append(doc_id)
        return index

    def _compute_idf(self) -> Dict[str, float]:
        """Computes inverse document frequency (IDF) for each term."""
        total_docs: int = len(self.documents)
        return {
            term: math.log((total_docs - len(doc_ids) + 0.5) /
                           (len(doc_ids) + 0.5) + 1)
            for term, doc_ids in self.inverted_index.items()
        }

    def _bm25_score(self, query_terms: List[str], doc: str, doc_id: int) -> float:
        """Computes BM25 score for a given document and query."""
        term_freqs: Counter = Counter(doc.split())
        doc_length: int = self.doc_lengths[doc_id]
        score: float = 0.0
        for term in query_terms:
            if term in self.idf:
                tf: int = term_freqs[term]
                numerator: float = tf * (self.k1 + 1)
                denominator: float = tf + self.k1 * \
                    (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += self.idf[term] * (numerator / denominator)
        return score

    def rerank(self, query: str, top_n: int = 5) -> List[Dict[str, float]]:
        """
        Rerank documents based on BM25 score for a given query.
        :param query: The search query
        :param top_n: Number of top results to return
        :return: List of dictionaries with "text" and "score"
        """
        query_terms: List[str] = query.split()
        scores: List[Tuple[int, float]] = [
            (doc_id, self._bm25_score(query_terms, doc, doc_id))
            for doc_id, doc in enumerate(self.documents)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        return [{"text": self.documents[doc_id], "score": score} for doc_id, score in scores[:top_n]]


# Functional Tests
if __name__ == "__main__":
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "The fast fox leaped over a sleeping dog",
        "A fox is a wild animal that is very quick",
        "Dogs and foxes can sometimes be friends",
        "The quick brown fox is very cunning and smart"
    ]
    reranker = BM25Reranker(documents)

    # Test 1: Query with common words
    query1 = "quick fox"
    result1 = reranker.rerank(query1, top_n=2)
    expected1_texts = [
        "The quick brown fox is very cunning and smart",
        "The quick brown fox jumps over the lazy dog"
    ]
    assert all(
        item["text"] in expected1_texts for item in result1), f"Test 1 Failed: {result1}"

    # Test 2: Query with rare terms
    query2 = "sleeping dog"
    result2 = reranker.rerank(query2, top_n=1)
    expected2_texts = [
        "The fast fox leaped over a sleeping dog"
    ]
    assert all(
        item["text"] in expected2_texts for item in result2), f"Test 2 Failed: {result2}"

    # Test 3: Query that matches multiple documents
    query3 = "fox"
    result3 = reranker.rerank(query3, top_n=3)
    expected3_texts = [
        "The fast fox leaped over a sleeping dog",
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox is very cunning and smart",
    ]
    assert all(
        item["text"] in expected3_texts for item in result3), f"Test 3 Failed: {result3}"

    print("All tests passed!")
