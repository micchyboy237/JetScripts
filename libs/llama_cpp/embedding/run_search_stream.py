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

    query = "Tell me about space exploration"

    documents = [
        # --- Space exploration (high relevance) ---
        "The Apollo program landed humans on the Moon in the late 1960s and early 1970s.",
        "Mars rovers like Curiosity and Perseverance search for signs of ancient microbial life.",
        "The James Webb Space Telescope studies the early universe using infrared observations.",
        "Black holes are regions of spacetime with gravity so strong that light cannot escape.",
        "The International Space Station supports long-term human habitation in orbit.",
        "SpaceX develops reusable rockets to reduce the cost of space travel.",
        "Saturn's rings are composed mostly of ice particles and rocky debris.",
        "Astronauts experience muscle loss and bone density reduction in microgravity.",
        # --- Adjacent science & technology (medium relevance) ---
        "Astrophysics combines physics and astronomy to study stars, galaxies, and the universe.",
        "Nuclear fusion powers stars by converting hydrogen into helium.",
        "Artificial intelligence is used to analyze large astronomical datasets.",
        "Telescopes can be ground-based or space-based depending on observation needs.",
        "Quantum mechanics describes physical phenomena at atomic scales.",
        # --- General technology / programming (low relevance) ---
        "Python is a popular programming language used in data science and machine learning.",
        "Distributed systems must handle network failures and partial outages.",
        "Docker allows applications to run in isolated containers.",
        "GPUs accelerate parallel computation for graphics and scientific workloads.",
        # --- Everyday / unrelated topics (noise) ---
        "Coffee contains caffeine, a stimulant that affects the central nervous system.",
        "Baking bread requires yeast, flour, water, and heat.",
        "Electric cars use lithium-ion batteries for energy storage.",
        "A healthy diet includes vegetables, fruits, and whole grains.",
        "Music streaming services recommend songs using collaborative filtering.",
    ]

    print(f"Streaming semantic search for: {query!r}\n")

    results_stream = embedder.search_stream(
        query=query,
        documents=documents,
        top_k=4,
        return_embeddings=False,
        batch_size=6,
        show_progress=True,
    )
    for r in results_stream:
        print(
            f"  • {r['score']:.4f}  {r['text'][:68]}{'...' if len(r['text']) > 68 else ''}"
        )

    print("Streaming complete.")


if __name__ == "__main__":
    main()
