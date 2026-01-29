#!/usr/bin/env python3
import os
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding


def main():
    embedder = LlamacppEmbedding(
        model="embeddinggemma",
        base_url=os.getenv("LLAMA_CPP_EMBED_URL"),
        use_cache=True,
        verbose=True,
    )

    documents = [
        "Machine learning models require large amounts of data.",
        "The capital of France is Paris.",
        "Photosynthesis is how plants make food.",
        "Bitcoin is a decentralized cryptocurrency.",
        "The mitochondria is the powerhouse of the cell.",
    ] * 20  # simulate larger set

    print("Streaming embeddings...\n")

    for i, batch in enumerate(
        embedder.get_embeddings_stream(
            documents,
            return_format="numpy",
            batch_size=8,
            show_progress=True,
        ),
        1,
    ):
        print(f"Batch {i:2d} received → {batch.shape[0]} embeddings")
        # You could already start using these embeddings here

    print("\nAll batches received.")


if __name__ == "__main__":
    main()
