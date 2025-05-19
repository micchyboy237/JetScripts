import os
from typing import List, Optional, Tuple, Union
from jet.file.utils import load_file
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics import silhouette_score


def cluster_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    reduce_dim: bool = True,
    n_components: int = 10,
    min_cluster_size: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Cluster a list of texts using Sentence Transformers, UMAP, and HDBSCAN.

    Args:
        texts (List[str]): List of texts to cluster.
        model_name (str): Sentence Transformer model name (default: "all-MiniLM-L6-v2").
        batch_size (int): Batch size for encoding (default: 32).
        device (str): Device for Sentence Transformer ("mps" for M1 Mac, "cpu", or "cuda").
        reduce_dim (bool): Whether to apply UMAP dimensionality reduction (default: True).
        n_components (int): Number of dimensions for UMAP reduction (default: 10).
        min_cluster_size (int): Minimum cluster size for HDBSCAN (default: 5).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - labels: Cluster labels (-1 for noise).
            - embeddings: Text embeddings (reduced if reduce_dim=True).
            - silhouette: Silhouette score for cluster quality (or -1 if <2 clusters).

    Raises:
        ValueError: If texts is empty or invalid parameters are provided.
    """
    if not texts:
        raise ValueError("Input text list cannot be empty.")

    # Step 1: Generate embeddings
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Step 2: Dimensionality reduction (optional)
    if reduce_dim:
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            metric="cosine",
            n_neighbors=15,
            min_dist=0.1
        )
        embeddings = reducer.fit_transform(embeddings)

    # Step 3: Cluster with HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean" if reduce_dim else "cosine",
        cluster_selection_method="eom",
        min_samples=None
    )
    labels = clusterer.fit_predict(embeddings)

    # Step 4: Compute silhouette score (if valid)
    silhouette = -1.0
    if len(np.unique(labels[labels >= 0])) >= 2:
        try:
            silhouette = silhouette_score(
                embeddings, labels, metric="euclidean" if reduce_dim else "cosine")
        except ValueError:
            pass

    return labels, embeddings, silhouette


# Example usage
if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers: List[str] = load_file(docs_file)
    sample_texts = [header["text"] for header in headers]

    # sample_texts = [
    #     "I love machine learning",
    #     "Deep learning is amazing",
    #     "Natural language processing is fun",
    #     "This is an unrelated topic",
    #     "Another distinct subject"
    # ]

    labels, embeddings, silhouette = cluster_texts(
        texts=sample_texts,
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        reduce_dim=True,
        n_components=5,
        min_cluster_size=2
    )

    # Print results
    for text, label in zip(sample_texts, labels):
        print(f"Text: {text} | Cluster: {label}")
    print(f"Silhouette Score: {silhouette:.4f}")
