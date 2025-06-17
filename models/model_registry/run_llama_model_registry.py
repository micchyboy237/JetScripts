from typing import Dict, Optional, TypedDict, Literal, Union
from jet.models.model_registry.transformers.llama_model_registry import LLaMAModelRegistry
import numpy as np
import logging

logger = logging.getLogger(__name__)


def main():
    """Example usage of LLaMAModelRegistry for text generation."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize registry
    registry = LLaMAModelRegistry()

    # Define model and features
    model_id = "meta-llama/Llama-2-7b"
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

        # Load tokenizer
        tokenizer = registry.get_tokenizer(model_id)
        if tokenizer is None:
            logger.error(f"Failed to load tokenizer for {model_id}")
            return

        # Example: Generate a response to a user query
        prompt = "What is the capital of France?"
        inputs = tokenizer(prompt, return_tensors="np" if isinstance(
            model, ONNXLLaMAWrapper) else "pt")
        input_ids = inputs["input_ids"]

        # Generate text
        max_length = 100
        if isinstance(model, ONNXLLaMAWrapper):
            generated_ids = model.generate(input_ids, max_length=max_length)
        else:
            generated_ids = model.generate(
                input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2)

        # Decode output
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        logger.info(f"Generated response: {output}")

    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")

    finally:
        # Clear registry to free resources
        registry.clear()


if __name__ == "__main__":
    main()
