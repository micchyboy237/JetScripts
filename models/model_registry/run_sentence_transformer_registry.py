from typing import Dict, Optional, TypedDict, Literal, Union
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def main():
    """Example usage of SentenceTransformerRegistry for semantic text similarity."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize registry
    registry = SentenceTransformerRegistry()

    # Define model and features
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    features: Dict[str, Literal["cpu", "cuda", "mps", "fp16", "fp32"]] = {
        "device": "mps",  # Optimized for Mac M1
        "precision": "fp16"
    }

    try:
        # Load model
        model = registry.load_model(model_id, features)
        if model is None:
            logger.error(f"Failed to load model {model_id}")
            return

        # Example: Compute semantic similarity between sentences
        sentences = [
            "The cat sits on the mat.",
            "A cat is resting on a rug.",
            "The dog runs in the park."
        ]

        # Encode sentences to get embeddings
        embeddings = model.encode(
            sentences, batch_size=32, convert_to_numpy=True)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")

        # Compute cosine similarity
        similarities = cosine_similarity(embeddings)
        logger.info("Cosine similarity matrix:")
        for i, sentence in enumerate(sentences):
            for j, other_sentence in enumerate(sentences):
                if i < j:
                    logger.info(
                        f"Similarity between '{sentence}' and '{other_sentence}': {similarities[i][j]:.4f}")

    except Exception as e:
        logger.error(f"Error during embedding generation: {str(e)}")

    finally:
        # Clear registry to free resources
        registry.clear()


if __name__ == "__main__":
    main()
