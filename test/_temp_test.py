# hybrid_search.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# Sample documents
documents = [
    "The cat is on the mat",
    "The dog is not on the mat",
    "A cat and a dog are friends",
    "The mat is clean",
    "The dog runs fast"
]

# Initialize models
dense_model = SentenceTransformer('all-MiniLM-L6-v2')
sparse_vectorizer = TfidfVectorizer()

# Precompute embeddings
dense_embeddings = dense_model.encode(documents, normalize_embeddings=True)
sparse_matrix = sparse_vectorizer.fit_transform(documents)


def hybrid_search(query, top_n=3, alpha=0.5):
    """
    alpha = 0.5 -> equally weight dense and sparse
    alpha closer to 1 -> favor dense (semantic)
    alpha closer to 0 -> favor sparse (exact)
    """
    # Dense embedding
    query_dense = dense_model.encode([query], normalize_embeddings=True)

    # Sparse vector
    query_sparse = sparse_vectorizer.transform([query])

    # Dense similarity
    dense_scores = util.cos_sim(query_dense, dense_embeddings)[0].cpu().numpy()

    # Sparse similarity
    sparse_scores = (query_sparse @ sparse_matrix.T).toarray()[0]

    # Hybrid score
    final_scores = alpha * dense_scores + (1 - alpha) * sparse_scores

    # Top N results
    top_indices = final_scores.argsort()[::-1][:top_n]
    results = [(documents[idx], final_scores[idx]) for idx in top_indices]

    return results


# Sample search
if __name__ == "__main__":
    query = "cat on mat"
    results = hybrid_search(query, top_n=3, alpha=0.5)

    print("\n🔎 Search Results:\n")
    for doc, score in results:
        print(f"Score: {score:.4f} | Document: {doc}")
