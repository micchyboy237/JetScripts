#!/usr/bin/env python3

import os
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding


def main():
    embedder = LlamacppEmbedding(
        model="embeddinggemma",  # or e5, nomic-embed, etc.
        base_url=os.getenv("LLAMA_CPP_EMBED_URL"),
        use_cache=True,
        use_dynamic_batch_sizing=True,
        verbose=True,
    )

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a great programming language.",
        "Coffee is the best morning drink.",
    ]

    print("Computing embeddings (single call)...")
    embeddings = embedder(
        texts,
        return_format="numpy",
        batch_size=512,
        show_progress=True,
    )

    print(f"→ Got {len(embeddings)} embeddings")
    print(f"→ Shape: {embeddings.shape}")
    print(f"→ First vector preview: {embeddings[0][:8]} ...")

    # Also works with a single string
    single_emb = embedder("Hello, world!")
    print(f"Single embedding shape: {single_emb.shape}")


if __name__ == "__main__":
    main()
