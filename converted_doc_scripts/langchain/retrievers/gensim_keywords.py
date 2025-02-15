from gensim.models import Word2Vec
import numpy as np


def search_word2vec(texts: list[str], query: str, model: Word2Vec, top_n: int = 5) -> list[tuple[str, float]]:
    """Search using Word2Vec embeddings and cosine similarity."""
    def get_embedding(text):
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    text_vectors = np.array([get_embedding(text) for text in texts])
    query_vector = get_embedding(query).reshape(1, -1)

    similarity_scores = cosine_similarity(query_vector, text_vectors).flatten()
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    return [(texts[i], similarity_scores[i]) for i in top_indices]


# Example Usage (After Training a Word2Vec Model)
sentences = [["machine", "learning", "is", "great"], ["deep", "learning",
                                                      "advances", "AI"], ["natural", "language", "processing", "is", "cool"]]
w2v_model = Word2Vec(sentences, vector_size=100, min_count=1, workers=4)

query = "AI and machine learning"
print(search_word2vec([" ".join(sent)
      for sent in sentences], query, w2v_model))
