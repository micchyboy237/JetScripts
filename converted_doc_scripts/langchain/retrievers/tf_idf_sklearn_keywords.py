from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def search_tfidf(texts: list[str], query: str, top_n: int = 5) -> list[tuple[str, float]]:
    """Search using TF-IDF with cosine similarity."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        texts + [query])  # Include query for comparison
    query_vector = tfidf_matrix[-1]  # Last vector is the query
    similarity_scores = cosine_similarity(
        query_vector, tfidf_matrix[:-1]).flatten()

    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    return [(texts[i], similarity_scores[i]) for i in top_indices]


# Example Usage
texts = ["Machine learning is great", "Deep learning advances AI",
         "Natural language processing is cool"]
query = "AI and machine learning"
print(search_tfidf(texts, query))
