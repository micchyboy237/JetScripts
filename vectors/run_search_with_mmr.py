import json
import os
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from jet.file.utils import load_file, save_file
from sklearn.utils import deprecation
from jet.vectors.search_with_mmr import search_diverse_context


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers = load_file(docs_file)
    # Remove headers with level 1
    headers = [header for header in headers if header["header_level"] != 1]
    query = "best Isekai anime 2024"

    results = search_diverse_context(
        query=query,
        headers=headers,
        model_name="all-MiniLM-L12-v2",
        rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        device="mps" if torch.backends.mps.is_available() else "cpu",
        top_k=20,
        num_results=10,
        lambda_param=0.5
    )

    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Text: {json.dumps(result['text'])[:100]}...")
        print(f"Embedding Score: {result['score']:.4f}")
        print(f"Rerank Score: {result['rerank_score']:.4f}")
        print("-" * 50)

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file(results, f"{output_dir}/search_with_mmr_results.json")
