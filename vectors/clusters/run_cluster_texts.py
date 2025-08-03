import json
import os
from typing import List, Optional, Tuple, Union, TypedDict
from jet.file.utils import load_file, save_file
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics import silhouette_score
from jet.vectors.clusters.base import cluster_texts

# Example usage
if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers: List[str] = load_file(docs_file)
    sample_texts = [header["text"] for header in headers]

    results = cluster_texts(
        texts=sample_texts,
        model_name="all-MiniLM-L12-v2",
        batch_size=32,
        device="cpu",
        reduce_dim=True,
        n_components=5,
        min_cluster_size=2
    )

    # Print results
    for result in results:
        print(f"Text: {json.dumps(result['text'])[:100]}")
        print(f"Cluster: {result['label']}")
        print(f"Probability: {result['cluster_probability']:.4f}")
        print(f"Is Noise: {result['is_noise']}")
        print(f"Cluster Size: {result['cluster_size']}")
        print("-" * 50)

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file(results, f"{output_dir}/clusters.json")
