from jet.llm.helpers.faiss_utils import create_faiss_index, get_faiss_model, search_faiss_index
import numpy as np


def main():
    dimension = 384  # MiniLM embedding size
    top_k = 3        # Retrieve top 3 results

    # Load Sentence Transformer Model
    model = get_faiss_model()

    # Corpus (Stored Sentences)
    corpus = [
        "Machine learning is transforming the world.",
        "Artificial intelligence is the future of technology.",
        "Python is a powerful programming language.",
        "Deep learning enables powerful AI models.",
        "FAISS is used for fast similarity search.",
        "NLP is a key part of AI research.",
        "Cloud computing allows scalable machine learning.",
        "Big data analytics improves business decision-making."
    ]

    # Encode Corpus into Embeddings
    corpus_embeddings = model.encode(
        corpus, normalize_embeddings=True).astype(np.float32)

    # Create FAISS Index
    faiss_index = create_faiss_index(corpus_embeddings, dimension)

    # User Query
    user_query = "What is the role of AI in technology?"
    query_embedding = model.encode(
        [user_query], normalize_embeddings=True).astype(np.float32)

    # Perform Search
    distances, indices = search_faiss_index(
        faiss_index, query_embedding, top_k)

    # Display Results
    print(f"\n🔹 Query: {user_query}")
    print("\n🔹 Top Matches:")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {corpus[idx]} (Score: {distances[0][i]:.4f})")


if __name__ == "__main__":
    main()
