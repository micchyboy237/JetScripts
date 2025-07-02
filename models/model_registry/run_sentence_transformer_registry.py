from typing import Union, List
import numpy as np

from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry

if __name__ == "__main__":
    # Initialize the registry
    registry = SentenceTransformerRegistry()

    # Load a model (sets model_id in the instance)
    model_id = "static-retrieval-mrl-en-v1"
    registry.load_model(
        model_id=model_id,
    )

    # Example 1: Generate embeddings for a single string
    single_input = "This is a sample sentence."
    embeddings_list = registry.generate_embeddings(
        input_data=single_input,
        batch_size=1,
        show_progress=False,
        return_format="list",
    )
    print("Single string embedding (list format):", embeddings_list)
    print("Embedding length:", len(embeddings_list))

    # Example 2: Generate embeddings for a list of strings
    multiple_inputs = [
        "This is the first sentence.",
        "This is another sentence for testing.",
        "A third sentence to embed."
    ]
    embeddings_numpy = registry.generate_embeddings(
        input_data=multiple_inputs,
        batch_size=2,
        show_progress=True,
        return_format="numpy",
    )
    print("Multiple strings embeddings (numpy format):", embeddings_numpy)
    print("Embeddings shape:", embeddings_numpy.shape)

    # Example 3: Attempt to generate embeddings without loading a model (will raise an error)
    new_registry = SentenceTransformerRegistry()
    try:
        new_registry.generate_embeddings(
            input_data="This will fail.",
            batch_size=1,
            return_format="list"
        )
    except ValueError as e:
        print("Expected error:", str(e))
