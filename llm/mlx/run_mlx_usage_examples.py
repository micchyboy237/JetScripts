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
    prompt = "Once upon a time"

    logit_bias = {tokenizer.encode(
        " magic", add_special_tokens=False)[0]: 10.0}
    output = mlx.generate(
        model=model,
        verbose=True,
        prompt=prompt,
        # temperature=0.7,  # Probabilistic sampling
        max_tokens=200,
        repetition_penalty=1.2,  # Mild penalty
        # repetition_context_size=20,
        # top_p=0.9,  # Nucleus sampling
        logit_bias=logit_bias
    )
    logger.info("Example 1 - Creative Writing:")
    logger.debug(f"logit_bias: {logit_bias}")
    logger.success(output["content"])
    logger.newline()


# Example 2: Chatbot with Greedy Decoding (Naive Detokenizer)
def example_chatbot_greedy():
    prompt = "Hello, how can I assist you today?"
    output = mlx.generate(
        model=model,
        verbose=True,
        prompt=prompt,
        temperature=0.0,  # Greedy decoding
        max_tokens=50,
        top_p=1.0,
        logit_bias=None
    )
    logger.info("Example 2 - Chatbot Greedy:")
    logger.success(output["content"])
    logger.newline()


# Example 3: Streaming Q&A with BPE Detokenizer (trim_space=False)
def example_streaming_qa():
    prompt = "What is the capital of France?"

    def formatter(text, prob):
        logger.info(f"[{prob:.2f}] {text}", end="")
    logger.info("Example 3 - Streaming Q&A:")
    for segment in mlx.stream_generate(
        model=model,
        verbose=True,
        prompt=prompt,
        temperature=0.5,
        max_tokens=30,
        # formatter=formatter,
        repetition_penalty=1.5,
        repetition_context_size=10,
        top_p=1.0,
        logit_bias=None
    ):
        pass
    logger.info("\n")


# Example 4: Text Completion with High Repetition Penalty (Naive Detokenizer)
def example_text_completion():
    prompt = "To install the software, first"
    output = mlx.generate(
        model=model,
        verbose=True,
        prompt=prompt,
        temperature=0.3,
        max_tokens=70,
        repetition_penalty=2.0,  # Strong penalty
        repetition_context_size=15,
        top_p=0.95,
        # Discourage "install"
        logit_bias={tokenizer.encode("install")[0]: -1.0}
    )
    logger.info("Example 4 - Text Completion:")
    logger.success(output["content"])
    logger.newline()


# Example 5: Streaming Creative Output with No Repetition Penalty (SPM Detokenizer, trim_space=False)
def example_streaming_creative():
    prompt = "The moon shines bright"

    def formatter(text, prob):
        logger.info(f"{text}", end="")
    logger.info("Example 5 - Streaming Creative:")
    for segment in mlx.stream_generate(
        model=model,
        verbose=True,
        prompt=prompt,
        temperature=0.8,
        max_tokens=50,
        # formatter=formatter,
        top_p=0.85,
        logit_bias={tokenizer.encode("light")[0]: 1.5}  # Bias toward "light"
    ):
        pass
    logger.info("\n")


# Example 6: Deterministic Short Answer with Logit Bias (BPE Detokenizer)
def example_short_answer():
    prompt = "The largest planet is"
    output = mlx.generate(
        model=model,
        verbose=True,
        prompt=prompt,
        temperature=0.0,  # Greedy
        max_tokens=10,
        repetition_penalty=1.1,
        repetition_context_size=5,
        top_p=1.0,
        # Strongly favor "Jupiter"
        logit_bias={tokenizer.encode("Jupiter")[0]: 5.0}
    )
    logger.info("Example 6 - Short Answer:")
    logger.success(output["content"])
    logger.newline()


# Example 7: Streaming Long-Form Content with Full Sampling (Naive Detokenizer)
def example_long_form_streaming():
    prompt = "Introduction to AI:"

    def formatter(text, prob):
        logger.info(f"{text}", end="")
    logger.info("Example 7 - Long-Form Streaming:")
    for segment in mlx.stream_generate(
        model=model,
        verbose=True,
        prompt=prompt,
        temperature=0.6,
        max_tokens=200,
        # formatter=formatter,
        repetition_penalty=1.3,
        repetition_context_size=50,
        top_p=0.9,
        logit_bias=None
    ):
        pass
    logger.info("\n")


if __name__ == "__main__":
    logger.info("Running MLX Language Model Generation Examples\n")
    example_creative_writing()
    example_chatbot_greedy()
    example_streaming_qa()
    example_text_completion()
    example_streaming_creative()
    example_short_answer()
    example_long_form_streaming()
    logger.info("All examples completed.")
