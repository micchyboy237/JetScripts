import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util


class HybridSearchEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2", diversity_penalty=0.3):
        self.dense_model = SentenceTransformer(model_name)
        self.sparse_vectorizer = TfidfVectorizer()
        self.documents = []
        self.dense_embeddings = None
        self.sparse_matrix = None
        self.diversity_penalty = diversity_penalty

    def fit(self, documents):
        """Fit the engine with documents."""
        self.documents = documents
        self.dense_embeddings = self.dense_model.encode(
            documents, normalize_embeddings=True)
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(documents)

    def search(self, query, top_n=5, alpha=0.5, diversity=True):
        """Search with a query using hybrid scoring."""
        if not self.documents:
            raise ValueError(
                "You must call `fit(documents)` before searching.")

        # Encode query
        query_dense = self.dense_model.encode(
            [query], normalize_embeddings=True)
        query_sparse = self.sparse_vectorizer.transform([query])

        # Dense similarity
        dense_scores = util.cos_sim(query_dense, self.dense_embeddings)[
            0].cpu().numpy()

        # Sparse similarity
        sparse_scores = (query_sparse @ self.sparse_matrix.T).toarray()[0]

        # Final hybrid score
        final_scores = alpha * dense_scores + (1 - alpha) * sparse_scores

        # Top N results before diversity
        top_indices = final_scores.argsort()[::-1][:top_n]

        if diversity:
            # Re-rank with diversity
            final_scores, top_indices = self.apply_diversity(
                top_indices, final_scores, top_n)

        # Sort by score in descending order
        sorted_indices = final_scores.argsort()[::-1]

        return [
            {
                "document": self.documents[idx],
                "score": float(final_scores[idx])
            }
            for idx in sorted_indices[:top_n]
        ]

    def apply_diversity(self, top_indices, final_scores, top_n):
        """Apply diversity penalty to the top N results."""
        selected_indices = []
        selected_scores = []
        selected_embeddings = []

        for idx in top_indices:
            # Check similarity with already selected results
            doc_embedding = self.dense_embeddings[idx]
            if not selected_embeddings:
                selected_indices.append(idx)
                selected_scores.append(final_scores[idx])
                selected_embeddings.append(doc_embedding)
            else:
                # Calculate cosine similarity with previous selections
                similarities = [util.cos_sim(doc_embedding, prev_emb)[
                    0] for prev_emb in selected_embeddings]
                min_similarity = min(similarities)

                if min_similarity < self.diversity_penalty:
                    # This document is sufficiently different, add it
                    selected_indices.append(idx)
                    selected_scores.append(final_scores[idx])
                    selected_embeddings.append(doc_embedding)

            # Ensure we stop if we have selected enough documents
            if len(selected_indices) >= top_n:
                break

        # If fewer documents are selected due to diversity, fill up with the remaining top ones
        remaining_indices = list(set(top_indices) - set(selected_indices))

        # Sort remaining indices by the highest final scores
        remaining_indices.sort(key=lambda idx: final_scores[idx], reverse=True)

        # Add remaining documents to meet the top_n requirement
        while len(selected_indices) < top_n and remaining_indices:
            selected_indices.append(remaining_indices.pop(0))

        # Ensure the final selection is sorted by score in descending order
        selected_indices_sorted = sorted(
            selected_indices, key=lambda idx: final_scores[idx], reverse=True)

        return np.array(final_scores)[selected_indices_sorted], np.array(selected_indices_sorted)


if __name__ == "__main__":
    docs = [
        "How to cook delicious pasta at home",
        "The art of boiling noodles perfectly",
        "Best practices for making pasta",
        "Top 10 pasta recipes you must try",
        "Healthy salad recipes with vegetables",
    ]
    query = "noodle cooking tips"

    engine = HybridSearchEngine()
    engine.fit(docs)

    results = engine.search(query, top_n=5, alpha=0.5)

    print("\n🔎 Hybrid Search Results w/ Diversity:\n")
    for r in results:
        print(f"Score: {r['score']:.4f} | Document: {r['document']}")

    results = engine.search(query, top_n=5, alpha=0.5, diversity=False)

    print("\n🔎 Hybrid Search Results w/o Diversity:\n")
    for r in results:
        print(f"Score: {r['score']:.4f} | Document: {r['document']}")
