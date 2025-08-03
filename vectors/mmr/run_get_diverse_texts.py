# JetScripts/vectors/mmr/run_get_diverse_texts.py
import os
import shutil
from typing import List
import numpy as np
from jet.file.utils import save_file
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.vectors.mmr import MMRResult, get_diverse_texts


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main() -> None:
    """
    Example usage of get_diverse_texts with real-world product descriptions.
    Uses SentenceTransformer to generate embeddings for a query and texts.
    """
    # Load SentenceTransformer model
    model = SentenceTransformerRegistry.load_model('all-MiniLM-L6-v2')

    # Simulated query and product descriptions
    query = "smartphone with good camera"
    texts = [
        "Phone A: 48MP camera, 6.5-inch AMOLED, 128GB storage",
        "Phone B: 50MP camera, 6.4-inch OLED, 256GB storage",
        "Phone C: 12MP camera, 6-inch LCD, budget-friendly",
        "Phone D: 16MP camera, 5000mAh battery, 6.2-inch display",
        "Phone E: 20MP camera, compact 5.8-inch display, 64GB storage",
    ]

    # Generate embeddings
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True)
    text_embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True)

    # Parameters
    mmr_lambda = 0.5  # Balance relevance and diversity
    num_results = 3   # Return top 3 diverse results
    initial_indices = [0]  # Pre-select Phone A as a starting point

    # Run MMR
    results: List[MMRResult] = get_diverse_texts(
        query_embedding=query_embedding,
        text_embeddings=text_embeddings,
        texts=texts,
        mmr_lambda=mmr_lambda,
        num_results=num_results,
        initial_indices=initial_indices
    )

    # Print results
    print("Query: 'smartphone with good camera'")
    print(f"MMR Lambda: {mmr_lambda}, Number of results: {num_results}")
    print("Selected diverse product descriptions:")
    for result in results:
        print(
            f"Index: {result['index']}, Similarity: {result['similarity']:.3f}")
        print(f"Text: {result['text']}")
        print()

    save_file(results, f"{OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
