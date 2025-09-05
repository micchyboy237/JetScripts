import os
from jet.llm.mlx.base import MLX
from jet.logger import logger
from transformers import AutoTokenizer
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate


model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
seed = 42
mlx = MLX(model, seed=seed)
tokenizer = mlx.tokenizer


# Example 1: Creative Writing with Probabilistic Sampling (SPM Detokenizer, trim_space=True)
def example_creative_writing():
    logger.info("Example 1 - Creative Writing:")

    prompt = "Once upon a time"

    output = mlx.generate(
        model=model,
        verbose=True,
        prompt=prompt,
        # temperature=0.7,  # Probabilistic sampling
        max_tokens=200,
        repetition_penalty=1.2,  # Mild penalty
        # repetition_context_size=20,
        # top_p=0.9,  # Nucleus sampling
        logit_bias={" magic": 10, "magic": 5}
    )

    logger.success(output["content"])
    logger.newline()


if __name__ == "__main__":
    logger.info("Running MLX Language Model Generation Examples\n")
    example_creative_writing()
