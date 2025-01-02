from functools import lru_cache
import sys
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Tuple
from jet.logger import time_it


def create_faiss_index(data: np.ndarray, dimension: int, nlist: int, metric: int = faiss.METRIC_INNER_PRODUCT) -> faiss.IndexIVFFlat:
    """
    Creates and trains a FAISS index with the given data.

    Parameters:
    - data: np.ndarray, the dataset to index.
    - dimension: int, the dimension of the data.
    - nlist: int, the number of clusters (nlist) for IVF index.
    - metric: FAISS metric type (default: faiss.METRIC_INNER_PRODUCT for cosine similarity).

    Returns:
    - index: Trained FAISS index of type IndexIVFFlat.
    """
    # Ensure nlist is not larger than the number of training points
    nlist = min(nlist, len(data))

    # Flat (brute-force) index for clustering
    quantizer = faiss.IndexFlatIP(dimension)  # Use cosine similarity
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric)

    # Train the index
    index.train(data)
    index.add(data)

    return index


def search_faiss_index(index: faiss.IndexIVFFlat, queries: np.ndarray, top_k: int, nprobe: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Searches the FAISS index for the nearest neighbors of the queries.

    Parameters:
    - index: FAISS index.
    - queries: np.ndarray, the query points.
    - top_k: int, the number of nearest neighbors to retrieve.
    - nprobe: int, the number of clusters to search (default: 1).

    Returns:
    - distances: np.ndarray, distances to the nearest neighbors.
    - indices: np.ndarray, indices of the nearest neighbors.
    """
    index.nprobe = nprobe
    distances, indices = index.search(queries, top_k)

    # Convert inner product to cosine similarity score (for better interpretation)
    distances = distances / np.linalg.norm(queries, axis=1)[:, None]
    return distances, indices


def get_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Convert bytes to MB or GB
def bytes_to_human_readable(byte_size):
    if byte_size < 1024:
        return f"{byte_size} bytes"
    elif byte_size < 1024 ** 2:
        return f"{byte_size / 1024:.2f} KB"
    elif byte_size < 1024 ** 3:
        return f"{byte_size / 1024**2:.2f} MB"
    else:
        return f"{byte_size / 1024**3:.2f} GB"


def main():
    model = time_it(get_model)()
    # Get memory size in a human-readable format
    model_size = sys.getsizeof(model)
    print(f"RAM usage of the model: {bytes_to_human_readable(model_size)}")

    candidates = [
        "The quick brown fox jumps over the lazy dog",
        "A fast fox jumped over a lazy dog",
        "Hello world, how are you?",
        "Data science and machine learning are fascinating",
        "Artificial intelligence is transforming the world"
    ]

    queries = [
        "What is artificial intelligence?",
        "Tell me about data science and machine learning",
        "What does the quick brown fox do?"
    ]

    candidate_embeddings = model.encode(candidates)
    query_embeddings = model.encode(queries)

    d = candidate_embeddings.shape[1]

    nlist = 100
    k = 3  # Number of nearest neighbors
    index = create_faiss_index(candidate_embeddings, d, nlist)

    distances, indices = search_faiss_index(
        index, query_embeddings, k, nprobe=10)

    # Display results with cosine similarity score
    print("Query Results (indices):")
    for i, query in enumerate(queries):
        print(f"\nQuery: {query}")
        for j in range(k):
            print(f"  Neighbor {
                  j + 1}: {candidates[indices[i][j]]} (Cosine Similarity: {distances[i][j]:.4f})")


if __name__ == "__main__":
    main()
