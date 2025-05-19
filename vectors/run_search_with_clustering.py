import json
import os
from typing import List, Optional, TypedDict, dict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from jet.file.utils import load_file, save_file
import umap
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.utils import deprecation

# Suppress warnings (from prior responses)
# os.environ["OMP_NESTED"] = "FALSE"


class ClusterResult(TypedDict):
    text: str
    label: int
    embedding: np.ndarray
    cluster_probability: float
    is_noise: bool
    cluster_size: int


def cluster_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L12-v2",
    batch_size: int = 32,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    reduce_dim: bool = True,
    n_components: int = 10,
    min_cluster_size: int = 5,
    random_state: int = 42
) -> List[ClusterResult]:
    """
    Cluster a list of texts using Sentence Transformers, UMAP, and HDBSCAN.

    Args:
        texts (List[str]): List of texts to cluster.
        model_name (str): Sentence Transformer model name (default: "all-MiniLM-L12-v2").
        batch_size (int): Batch size for encoding (default: 32).
        device (str): Device for Sentence Transformer ("mps" for M1 Mac, "cpu", or "cuda").
        reduce_dim (bool): Whether to apply UMAP dimensionality reduction (default: True).
        n_components (int): Number of dimensions for UMAP reduction (default: 10).
        min_cluster_size (int): Minimum cluster size for HDBSCAN (default: 5).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        List[ClusterResult]: List of dictionaries containing clustering results with:
            - text: Original input text
            - label: Cluster label (-1 for noise)
            - embedding: Text embedding (reduced if reduce_dim=True)
            - cluster_probability: HDBSCAN membership probability
            - is_noise: Whether the point is classified as noise
            - cluster_size: Number of points in the assigned cluster

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
        min_samples=None,
        prediction_data=True
    )
    labels = clusterer.fit_predict(embeddings)
    probabilities = clusterer.probabilities_

    # Step 4: Compute cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))

    # Step 5: Build results
    results: List[ClusterResult] = []
    for i, text in enumerate(texts):
        label = int(labels[i])
        result: ClusterResult = {
            "text": text,
            "label": label,
            "embedding": embeddings[i],
            "cluster_probability": float(probabilities[i]),
            "is_noise": label == -1,
            "cluster_size": int(cluster_sizes.get(label, 0))
        }
        results.append(result)

    return results


def preprocess_texts(headers: List[dict]) -> List[str]:
    """
    Filter out noisy texts (e.g., menus, short texts) from headers.
    Args:
        headers: List of header dicts with 'text' key.
    Returns:
        List of cleaned texts.
    """
    exclude_keywords = ["menu", "sign in", "trending", "close"]
    min_words = 10
    return [
        header["text"]
        for header in headers
        if not any(keyword in header["text"].lower() for keyword in exclude_keywords)
        and len(header["text"].split()) >= min_words
    ]


def embed_search(
    query: str,
    texts: List[str],
    model_name: str = "all-MiniLM-L12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    top_k: int = 20
) -> List[dict]:
    """
    Perform embedding-based search to retrieve top-k relevant texts.
    Args:
        query: Search query.
        texts: List of corpus texts.
        model_name: Sentence Transformer model.
        device: Device for encoding (mps for M1).
        top_k: Number of candidates to retrieve.
    Returns:
        List of dicts with text, score, and embedding.
    """
    model = SentenceTransformer(model_name, device=device)
    query_embedding = model.encode(
        query, convert_to_tensor=True, device=device)
    text_embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device
    )
    similarities = util.cos_sim(query_embedding, text_embeddings)[
        0].cpu().numpy()
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    return [
        {
            "text": texts[i],
            "score": float(similarities[i]),
            "embedding": text_embeddings[i].cpu().numpy()
        }
        for i in top_k_indices
    ]


def rerank_results(
    query: str,
    candidates: List[dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
) -> List[dict]:
    """
    Rerank candidates using a cross-encoder.
    Args:
        query: Search query.
        candidates: List of candidate dicts with 'text' and 'score'.
        model_name: Cross-encoder model.
        device: Device for encoding.
    Returns:
        Reranked list of dicts with updated scores.
    """
    model = CrossEncoder(model_name, device=device)
    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = model.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


def cluster_diversity(
    candidates: List[dict],
    num_results: int = 5,
    model_name: str = "all-MiniLM-L12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    min_cluster_size: int = 2,
    n_components: int = 5
) -> List[dict]:
    """
    Select diverse results by clustering candidates using cluster_texts.
    Args:
        candidates: List of candidate dicts with 'text', 'score', 'embedding'.
        num_results: Number of final results.
        model_name: Sentence Transformer model (for cluster_texts).
        device: Device for clustering.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        n_components: UMAP dimensions.
    Returns:
        List of diverse results with cluster labels.
    """
    if not candidates:
        return []

    # Extract texts for clustering
    texts = [candidate["text"] for candidate in candidates]

    # Cluster candidates
    cluster_results = cluster_texts(
        texts=texts,
        model_name=model_name,
        batch_size=32,
        device=device,
        reduce_dim=True,
        n_components=n_components,
        min_cluster_size=min_cluster_size,
        random_state=42
    )

    # Group candidates by cluster label, excluding noise
    cluster_groups = {}
    for cluster_result, candidate in zip(cluster_results, candidates):
        label = cluster_result["label"]
        if not cluster_result["is_noise"]:
            if label not in cluster_groups:
                cluster_groups[label] = []
            candidate["cluster_label"] = label
            cluster_groups[label].append(candidate)

    # Select highest rerank_score from each cluster
    selected = []
    for label in sorted(cluster_groups.keys()):
        cluster_candidates = sorted(
            cluster_groups[label],
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        selected.append(cluster_candidates[0])  # Take top-scoring candidate
        if len(selected) >= num_results:
            break

    return selected[:num_results]


def search_diverse_context(
    query: str,
    headers: List[dict],
    model_name: str = "all-MiniLM-L12-v2",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    top_k: int = 20,
    num_results: int = 5,
    min_cluster_size: int = 2,
    n_components: int = 5
) -> List[dict]:
    """
    Search for diverse context data using embedding search, reranking, and clustering.
    Args:
        query: Search query.
        headers: List of header dicts with 'text'.
        model_name: Sentence Transformer model.
        rerank_model: Cross-encoder model.
        device: Device for encoding.
        top_k: Number of candidates for reranking.
        num_results: Number of final diverse results.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        n_components: UMAP dimensions for clustering.
    Returns:
        List of dicts with text, score, rerank_score, embedding, and cluster_label.
    """
    # Preprocess texts
    texts = preprocess_texts(headers)
    if not texts:
        return []

    # Embedding-based search
    candidates = embed_search(query, texts, model_name, device, top_k)

    # Rerank
    reranked = rerank_results(query, candidates, rerank_model, device)

    # Diversity via clustering
    diverse_results = cluster_diversity(
        reranked,
        num_results,
        model_name,
        device,
        min_cluster_size,
        n_components
    )

    return diverse_results


# Example usage
if __name__ == "__main__":
    # Load data
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers = load_file(docs_file)

    # Query
    query = "best Isekai anime 2024"

    # Search
    results = search_diverse_context(
        query=query,
        headers=headers,
        model_name="all-MiniLM-L12-v2",
        rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        device="mps" if torch.backends.mps.is_available() else "cpu",
        top_k=20,
        num_results=5,
        min_cluster_size=2,
        n_components=5
    )

    # Print results
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Text: {json.dumps(result['text'])[:100]}...")
        print(f"Embedding Score: {result['score']:.4f}")
        print(f"Rerank Score: {result['rerank_score']:.4f}")
        print(f"Cluster Label: {result['cluster_label']}")
        print("-" * 50)

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file(results, f"{output_dir}/search_with_clustering_results.json")
