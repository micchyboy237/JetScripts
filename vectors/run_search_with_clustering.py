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
from jet.vectors.search_with_clustering import search_documents


# Example usage
if __name__ == "__main__":
    # Load data
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers = load_file(docs_file)

    # Query
    query = "best Isekai anime 2024"

    # Search
    results = search_documents(
        query=query,
        headers=headers,
        model_name="all-MiniLM-L12-v2",
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
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
