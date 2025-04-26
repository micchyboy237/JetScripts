import os
from transformers import AutoTokenizer
import mlx.core as mx
import mlx.nn as nn
from llama_index.llms.mlx.utils import gen_full, gen_stream
from llama_index.llms.mlx.tokenizer_utils import load_tokenizer
from mlx_lm import load, generate


model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
model, tokenizer = load(model_path)


# Example 1: Chatbot with Greedy Decoding (Naive Detokenizer)
def example_chatbot_greedy():
    prompt = "Hello, how can I assist you today?"
    output = gen_full(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temp=0.0,  # Greedy decoding
        max_tokens=50,
        repetition_penalty=None,
        repetition_context_size=None,
        top_p=1.0,
        logit_bias=None
    )
    print("Example 1 - Chatbot Greedy:")
    print(output)
    print()


# Example 2: Creative Writing with Probabilistic Sampling (SPM Detokenizer, trim_space=True)
def example_creative_writing():
    prompt = "Once upon a time"

    output = gen_full(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temp=0.7,  # Probabilistic sampling
        max_tokens=100,
        repetition_penalty=1.2,  # Mild penalty
        repetition_context_size=20,
        top_p=0.9,  # Nucleus sampling
        logit_bias={tokenizer.encode("magic")[0]: 2.0}  # Bias toward "magic"
    )
    print("Example 2 - Creative Writing:")
    print(output)
    print()


# Example 3: Streaming Q&A with BPE Detokenizer (trim_space=False)
def example_streaming_qa():
    prompt = "What is the capital of France?"

    def formatter(text, prob):
        print(f"[{prob:.2f}] {text}", end="")
    print("Example 3 - Streaming Q&A:")
    for segment in gen_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temp=0.5,
        max_tokens=30,
        formatter=formatter,
        repetition_penalty=1.5,
        repetition_context_size=10,
        top_p=1.0,
        logit_bias=None
    ):
        pass
    print("\n")


# Example 4: Text Completion with High Repetition Penalty (Naive Detokenizer)
def example_text_completion():
    prompt = "To install the software, first"
    output = gen_full(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temp=0.3,
        max_tokens=70,
        repetition_penalty=2.0,  # Strong penalty
        repetition_context_size=15,
        top_p=0.95,
        # Discourage "install"
        logit_bias={tokenizer.encode("install")[0]: -1.0}
    )
    print("Example 4 - Text Completion:")
    print(output)
    print()


# Example 5: Streaming Creative Output with No Repetition Penalty (SPM Detokenizer, trim_space=False)
def example_streaming_creative():
    prompt = "The moon shines bright"

    def formatter(text, prob):
        print(f"{text}", end="")
    print("Example 5 - Streaming Creative:")
    for segment in gen_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temp=0.8,
        max_tokens=50,
        formatter=formatter,
        repetition_penalty=None,
        repetition_context_size=None,
        top_p=0.85,
        logit_bias={tokenizer.encode("light")[0]: 1.5}  # Bias toward "light"
    ):
        print(segment, end='', flush=True)
    print("\n")


# Example 6: Deterministic Short Answer with Logit Bias (BPE Detokenizer)
def example_short_answer():
    prompt = "The largest planet is"
    output = gen_full(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temp=0.0,  # Greedy
        max_tokens=10,
        repetition_penalty=1.1,
        repetition_context_size=5,
        top_p=1.0,
        # Strongly favor "Jupiter"
        logit_bias={tokenizer.encode("Jupiter")[0]: 5.0}
    )
    print("Example 6 - Short Answer:")
    print(output)
    print()


# Example 7: Streaming Long-Form Content with Full Sampling (Naive Detokenizer)
def example_long_form_streaming():
    prompt = "Introduction to AI:"

    def formatter(text, prob):
        print(f"{text}", end="")
    print("Example 7 - Long-Form Streaming:")
    for segment in gen_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temp=0.6,
        max_tokens=200,
        formatter=formatter,
        repetition_penalty=1.3,
        repetition_context_size=50,
        top_p=0.9,
        logit_bias=None
    ):
        pass
    print("\n")


def main():
    print("Running MLX Language Model Generation Examples\n")
    example_chatbot_greedy()
    example_creative_writing()
    example_streaming_qa()
    example_text_completion()
    example_streaming_creative()
    example_short_answer()
    example_long_form_streaming()
    print("All examples completed.")


if __name__ == "__main__":
    main()
