import faiss


def search_faiss(texts: list[str], query: str, model_name: str = 'all-MiniLM-L6-v2', top_n: int = 5):
    """Search using FAISS for fast similarity search."""
    model = SentenceTransformer(model_name)
    text_vectors = model.encode(texts).astype('float32')

    d = text_vectors.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(text_vectors)

    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, top_n)

    return [(texts[i], 1 - distances[0][j]) for j, i in enumerate(indices[0])]


# Example Usage
texts = ["Machine learning is great", "Deep learning advances AI",
         "Natural language processing is cool"]
query = "AI and machine learning"
print(search_faiss(texts, query))
