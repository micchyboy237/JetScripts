from sentence_transformers import SentenceTransformer


def search_sbert(texts: list[str], query: str, model_name: str = 'all-MiniLM-L6-v2', top_n: int = 5) -> list[tuple[str, float]]:
    """Search using SBERT embeddings and cosine similarity."""
    model = SentenceTransformer(model_name)
    text_vectors = model.encode(texts, convert_to_tensor=True)
    query_vector = model.encode(query, convert_to_tensor=True)

    similarity_scores = cosine_similarity(query_vector.cpu().numpy().reshape(
        1, -1), text_vectors.cpu().numpy()).flatten()
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    return [(texts[i], similarity_scores[i]) for i in top_indices]


# Example Usage
texts = ["Machine learning is great", "Deep learning advances AI",
         "Natural language processing is cool"]
query = "AI and machine learning"
print(search_sbert(texts, query))
