from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import torch


def load_models():
    """Load sentence transformer and cross-encoder models."""
    # Sentence Transformer for initial retrieval
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Cross-Encoder for reranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    return embedder, cross_encoder


def create_index(embedder, documents):
    """Create FAISS index with document embeddings."""
    # Encode documents
    doc_embeddings = embedder.encode(documents, convert_to_tensor=False)
    # Initialize FAISS index (L2 distance)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    # Add embeddings to index
    index.add(doc_embeddings)
    return index, doc_embeddings


def retrieve_candidates(embedder, index, query, k=10):
    """Retrieve top-k candidates using sentence transformer embeddings."""
    # Encode query
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    # Search index
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]


def rerank_candidates(cross_encoder, query, documents, candidate_indices):
    """Rerank candidates using cross-encoder."""
    # Prepare query-document pairs
    pairs = [[query, documents[idx]] for idx in candidate_indices]
    # Score pairs with cross-encoder
    scores = cross_encoder.predict(pairs)
    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    return candidate_indices[sorted_indices], scores[sorted_indices]


def main():
    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fox fled from danger in the forest.",
        "Dogs are loyal and friendly pets.",
        "The cat sleeps on the windowsill.",
        "Foxes are known for their cunning behavior."
    ]

    # Sample query
    query = "Tell me about foxes."

    # Load models
    print("Loading models...")
    embedder, cross_encoder = load_models()

    # Create FAISS index
    print("Creating FAISS index...")
    index, doc_embeddings = create_index(embedder, documents)

    # Step 1: Retrieve candidates
    print(f"\nRetrieving candidates for query: '{query}'")
    candidate_indices, candidate_distances = retrieve_candidates(
        embedder, index, query, k=3)
    print("Initial retrieval results:")
    for idx, dist in zip(candidate_indices, candidate_distances):
        print(f"Doc {idx}: {documents[idx]} (Distance: {dist:.4f})")

    # Step 2: Rerank candidates
    print("\nReranking candidates...")
    reranked_indices, reranked_scores = rerank_candidates(
        cross_encoder, query, documents, candidate_indices)
    print("Reranked results:")
    for idx, score in zip(reranked_indices, reranked_scores):
        print(f"Doc {idx}: {documents[idx]} (Score: {score:.4f})")


if __name__ == "__main__":
    main()
