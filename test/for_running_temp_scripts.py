from sklearn.cluster import KMeans
from typing import List, Dict, Callable, Optional, TypedDict
from sklearn.metrics.pairwise import cosine_similarity
from jet.llm.utils.embeddings import SFEmbeddingFunction
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Callable, List


def plot_text_embeddings(texts: List[str], embeddings: List[List[float]], title: str = "Text Embeddings Visualization"):
    """
    Plots text embeddings in a 2D space using PCA and auto-opens the viewer.

    Args:
        texts (List[str]): List of text inputs.
        embeddings (List[List[float]]): Corresponding embeddings.
        title (str): Title of the plot.
    """
    if len(embeddings) == 0 or len(embeddings[0]) < 2:
        raise ValueError(
            "Embeddings must have at least two dimensions for visualization.")

    # Reduce embeddings to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 7))
    plt.scatter(reduced_embeddings[:, 0],
                reduced_embeddings[:, 1], marker='o', alpha=0.7)

    # Annotate points with corresponding text (truncate long texts for readability)
    for i, text in enumerate(texts):
        plt.annotate(text[:20] + '...' if len(text) > 20 else text,
                     (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                     fontsize=9, alpha=0.75)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.grid(True)

    # Automatically open the viewer
    plt.show()


def cluster_texts(
    texts: List[str],
    embedding_function: Callable[[List[str]], List[List[float]]],
    num_clusters: Optional[int] = None
) -> Dict[int, List[str]]:
    """
    Groups similar texts into clusters based on embeddings.

    Args:
        texts (List[str]): List of text inputs.
        embedding_function (Callable): Function to generate text embeddings.
        num_clusters (int, optional): Number of clusters. If None, it will be auto-determined.

    Returns:
        Dict[int, List[str]]: Dictionary mapping cluster IDs to lists of similar texts.
    """

    # Generate embeddings
    embeddings = np.array(embedding_function(texts))

    # Auto-determine number of clusters if not provided
    if num_clusters is None:
        num_clusters = max(2, min(len(texts) // 3, 10))  # Dynamic selection

    # Clustering using KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Organize texts into clusters
    clustered_texts = {}
    for i, label in enumerate(cluster_labels):
        clustered_texts.setdefault(label, []).append(texts[i])

    return clustered_texts


class SimilarResult(TypedDict):
    text: str
    score: float


def find_most_similar_texts(
    texts: List[str],
    embedding_function: Callable[[List[str]], List[List[float]]],
    *,
    threshold: float = 0.25,
    num_decimal: int = 2
) -> Dict[str, List[SimilarResult]]:
    """
    Finds the most similar texts using cosine similarity.

    Args:
        texts (List[str]): List of text inputs.
        embedding_function (Callable): Function to generate text embeddings.
        threshold (float): Similarity threshold for grouping.
        num_decimal (int): Number of decimal places to truncate similarity scores.

    Returns:
        Dict[str, List[SimilarResult]]: Dictionary mapping each text to similar ones with scores.
    """
    embeddings = np.array(embedding_function(texts))
    similarity_matrix = cosine_similarity(embeddings)

    factor = 10 ** num_decimal  # Scaling factor for truncation

    text_with_scores = {}
    for i, text in enumerate(texts):
        score_results = [
            {"text": texts[j], "score": similarity_matrix[i, j]}
            for j in range(len(texts))
            # Truncate without rounding
            if (int(similarity_matrix[i, j] * factor) / factor) > threshold and i != j
        ]

        text_with_scores[text] = sorted(
            score_results, key=lambda x: x["score"], reverse=True
        )

    return text_with_scores


if __name__ == "__main__":
    # Sample texts with varying similarities and differences
    texts = [
        # Group 1: Technology
        "Artificial Intelligence is transforming industries.",
        "Machine Learning models predict outcomes using data.",
        "Deep Learning is a subset of machine learning.",
        "Neural networks simulate the human brain.",

        # Group 2: Space and Astronomy
        "NASA discovered a new exoplanet in the habitable zone.",
        "Black holes warp space-time due to their gravity.",
        "The James Webb Telescope captures deep-space images.",
        "Astrobiology explores the potential for extraterrestrial life.",

        # Group 3: Sports
        "Soccer is the world's most popular sport.",
        "Basketball requires agility and teamwork.",
        "Tennis matches can last for hours in Grand Slams.",
        "Formula 1 cars are designed for maximum speed and aerodynamics.",

        # Group 4: Nature & Environment
        "Climate change is affecting global weather patterns.",
        "Deforestation leads to habitat loss and species extinction.",
        "Renewable energy sources include solar and wind power.",
        "Oceans absorb a large percentage of the Earth's heat.",

        # Group 5: Random (for diversity)
        "Cooking is an art that blends flavors and techniques.",
        "Music has the power to evoke emotions and memories.",
        "Philosophy questions the nature of existence and reality.",
        "History teaches us lessons from past civilizations."
    ]

    # Generate embeddings (Replace with actual embedding function)
    embedding_function = SFEmbeddingFunction("paraphrase-MiniLM-L12-v2")
    embeddings = embedding_function(texts)

    # Plot the embeddings
    plot_text_embeddings(texts, embeddings)

    print("Done")
