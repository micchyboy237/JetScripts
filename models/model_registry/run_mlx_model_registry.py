from typing import Dict, Optional, TypedDict, Literal, Union
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
import numpy as np
import logging
import mlx.core as mx

logger = logging.getLogger(__name__)


def main():
    """Demonstrate MLXModelRegistry usage with real-world examples."""
    registry = MLXModelRegistry()
    model_id = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
    features: Dict[str, Literal["cpu", "mps", "fp16", "fp32"]] = {
        "device": "mps",
        "precision": "fp16"
    }

    try:
        # Example 1: Text Generation
        logger.info("Example 1: Generating text with MLX model")
        model = registry.load_model(model_id, features)
        if model is None:
            logger.error(f"Failed to load model {model_id}")
            return

        tokenizer = registry.get_tokenizer(model_id)
        if tokenizer is None:
            logger.error(f"Failed to load tokenizer for {model_id}")
            return

        # Text generation prompt
        prompt = "Write a short poem about the night sky:"
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        max_length = 100

        # Generate text
        model_inputs = mx.array(input_ids, dtype=mx.int32)
        generated_ids = model(model_inputs)
        if isinstance(generated_ids, mx.array):
            generated_ids = np.array(generated_ids)
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        logger.info(f"Generated poem: {output}")

        # Example 2: Text Classification
        logger.info(
            "\nExample 2: Performing text classification with MLX model")
        prompt = (
            "Classify the sentiment of this review as positive or negative:\n"
            "The movie was incredibly engaging with stunning visuals!"
        )
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]

        model_inputs = mx.array(input_ids, dtype=mx.int32)
        logits = model(model_inputs)
        if isinstance(logits, mx.array):
            logits = np.array(logits)

        # Simple sentiment classification based on logits
        sentiment = "positive" if logits[0][-1] > 0 else "negative"
        logger.info(f"Review sentiment: {sentiment}")

        # Example 3: Question Answering
        logger.info("\nExample 3: Answering a question with MLX model")
        prompt = (
            "Context: The Eiffel Tower is located in Paris, France.\n"
            "Question: Where is the Eiffel Tower located?"
        )
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]

        model_inputs = mx.array(input_ids, dtype=mx.int32)
        generated_ids = model(model_inputs)
        if isinstance(generated_ids, mx.array):
            generated_ids = np.array(generated_ids)
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        logger.info(f"Answer: {output}")

    except Exception as e:
        logger.error(f"Error during MLX model operations: {str(e)}")

    finally:
        registry.clear()


if __name__ == "__main__":
    main()
