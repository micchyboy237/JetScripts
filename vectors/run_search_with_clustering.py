import json
import os
from typing import List, Optional, TypedDict
from jet.vectors.cluster import cluster_texts
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from jet.file.utils import load_file, save_file
import umap
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.utils import deprecation


def preprocess_texts(headers: List[dict]) -> List[str]:
    """
    Filter out noisy texts (e.g., menus, short texts) from headers.
    Args:
        headers: List of header dicts with 'text' key.
    Returns:
        List of cleaned texts.
    """
    return [header["text"] for header in headers]


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


def cluster_texts_initial(
    texts: List[str],
    model_name: str = "all-MiniLM-L12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    min_cluster_size: int = 2,
    n_components: int = 5
) -> List[dict]:
    """
    Cluster texts using cluster_texts function.
    Args:
        texts: List of texts to cluster.
        model_name: Sentence Transformer model.
        device: Device for clustering.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        n_components: UMAP dimensions.
    Returns:
        List of dicts with text, cluster label, and embedding.
    """
    cluster_results = cluster_texts(
        texts=texts,
        model_name=model_name,
        batch_size=32,
        device=device,
        reduce_dim=True,
        n_components=n_components,
        min_cluster_size=min_cluster_size,
    )
    return [
        {
            "text": result["text"],
            "cluster_label": result["label"],
            # Convert to list for JSON serialization
            "embedding": result["embedding"].tolist(),
            "is_noise": result["is_noise"]
        }
        for result in cluster_results
    ]


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
    Search for diverse context data by clustering texts first, then applying search logic.
    Args:
        query: Search query.
        headers: List of header dicts with 'text'.
        model_name: Sentence Transformer model.
        rerank_model: Cross-encoder model.
        device: Device for encoding.
        top_k: Number of candidates for reranking per cluster.
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

    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    os.makedirs(output_dir, exist_ok=True)

    # Cluster texts first
    clustered_texts = cluster_texts_initial(
        texts,
        model_name,
        device,
        min_cluster_size,
        n_components
    )

    # Save clusters to a separate file
    clusters_file = os.path.join(output_dir, "clusters.json")
    save_file(clustered_texts, clusters_file)

    # Group texts by cluster
    cluster_groups = {}
    for item in clustered_texts:
        if item["is_noise"]:
            continue
        label = item["cluster_label"]
        if label not in cluster_groups:
            cluster_groups[label] = []
        cluster_groups[label].append(item)

    # Perform search within each cluster
    selected_candidates = []
    for label, cluster_items in cluster_groups.items():
        cluster_texts = [item["text"] for item in cluster_items]
        # Embedding-based search within cluster
        candidates = embed_search(
            query, cluster_texts, model_name, device, top_k)
        # Add cluster label and original embedding
        for candidate in candidates:
            # Find original clustered text to get cluster_label and embedding
            original_item = next(
                (item for item in cluster_items if item["text"]
                 == candidate["text"]),
                None
            )
            if original_item:
                candidate["cluster_label"] = original_item["cluster_label"]
                candidate["embedding"] = np.array(
                    original_item["embedding"])  # Convert back to numpy array

        # Rerank candidates within cluster
        reranked = rerank_results(query, candidates, rerank_model, device)
        # Select top candidate from cluster
        if reranked:
            selected_candidates.append(reranked[0])

    # Sort by rerank_score and select top num_results
    selected_candidates = sorted(
        selected_candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )[:num_results]

    return selected_candidates


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
