"""
Demo 2: SPLADE via FastEmbed (ONNX-optimized)
Model: prithivida/Splade_PP_en_v1
Strategy: High-throughput sparse encoding with indices/values format
"""

import numpy as np
from fastembed import SparseTextEmbedding


def main():
    # 1. Load model (auto-downloads ONNX weights)
    model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    documents = [
        "Apple releases new iPhone with titanium frame",
        "The weather is lovely and sunny today",
        "Chandrayaan-3 landed on the Moon in August 2023",
    ]

    # 2. Batch encode → list of SparseEmbedding objects
    sparse_embeddings = list(model.embed(documents, batch_size=4))

    # 3. Inspect raw output format (indices + values arrays)
    for i, emb in enumerate(sparse_embeddings):
        print(f"[{i}] {documents[i]}")
        print(f"    Active dimensions: {len(emb.indices)}")
        print(f"    Indices (vocab IDs): {emb.indices[:8]}...")
        print(f"    Values  (weights):   {np.round(emb.values[:8], 3)}...")
        print()

    # 4. Convert to dict format compatible with vector DBs
    # Many DBs (Qdrant, Pinecone) accept {index: weight} dicts
    db_ready = [
        {int(idx): float(val) for idx, val in zip(emb.indices, emb.values)}
        for emb in sparse_embeddings
    ]
    print("DB-ready format (first doc, first 5 entries):")
    print(dict(list(db_ready[0].items())[:5]))


if __name__ == "__main__":
    main()
