from unittest.mock import MagicMock
from jet.file.utils import save_file
from jet.llm.mlx.generation import generate
import pytest
import os
from typing import Union, Dict, List, TypedDict
from jet.llm.mlx.mlx_types import CompletionResponse, LLMModelType
from jet.llm.mlx.token_utils import get_tokenizer
from transformers import PreTrainedTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper
import math

MLXTokenizer = Union[TokenizerWrapper, PreTrainedTokenizer]


def read_logprobs(
    completion_response: CompletionResponse,
    top_n: int = 5
) -> List[Dict[str, Union[int, str, float, List[Dict[str, Union[int, str, float]]]]]]:
    """
    Reads and processes logprobs from a CompletionResponse, converting token IDs to strings,
    adding confidence scores, and generating full alternative sequences.

    Args:
        completion_response: CompletionResponse containing choices with logprobs.
        top_n: Maximum number of top logprobs to include for each token (default: 5).

    Returns:
        Dictionary containing:
        - tokens: List of dictionaries with:
            - token_id: The token ID.
            - token_str: The decoded token string.
            - logprob: The log probability of the token.
            - confidence: The probability (exp(logprob)) of the token.
            - top_logprobs: List of top_n logprobs with token_id, token_str, logprob, and confidence.
        - sequences: List of sequences (number based on max entries in top_logprobs, up to top_n) with their average confidence.
    """
    if not completion_response["choices"] or not completion_response["choices"][0].get("logprobs"):
        return {"tokens": [], "sequences": []}

    model: LLMModelType = completion_response["model"]
    tokenizer: MLXTokenizer = get_tokenizer(model)

    logprobs = completion_response["choices"][0]["logprobs"]
    tokens = logprobs["tokens"]
    token_logprobs = logprobs["token_logprobs"]
    top_logprobs = logprobs["top_logprobs"]

    # Process tokens with confidence scores
    token_results = []
    max_top_n = 0
    for token_id, logprob, top_logprob_dict in zip(tokens, token_logprobs, top_logprobs):
        # Decode token ID to string
        token_str = tokenizer.decode(
            [token_id]) if token_id is not None else ""
        confidence = math.exp(logprob) if logprob is not None else 0.0

        # Process top_logprobs, sorting by logprob descending
        sorted_top = sorted(
            top_logprob_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        max_top_n = max(max_top_n, len(top_logprob_dict))

        top_logprobs_list = [
            {
                "token_id": int(t_id),
                "token_str": tokenizer.decode([int(t_id)]) if t_id is not None else "",
                "logprob": logprob,
                "confidence": math.exp(logprob) if logprob is not None else 0.0
            }
            for t_id, logprob in sorted_top
        ]

        token_results.append({
            "token_id": token_id,
            "token_str": token_str,
            "logprob": logprob,
            "confidence": confidence,
            "top_logprobs": top_logprobs_list
        })

    # Determine number of sequences as max entries in top_logprobs, capped at top_n
    num_sequences = min(max_top_n, top_n)

    # Generate sequences
    sequences = []
    for i in range(num_sequences):
        seq_tokens = []
        seq_confidences = []
        for j, top_logprob_dict in enumerate(top_logprobs):
            sorted_top = sorted(
                top_logprob_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
            # Select the i-th top token, or the last one if i exceeds available tokens
            idx = min(i, len(sorted_top) - 1)
            token_id, logprob = sorted_top[idx]
            seq_tokens.append(int(token_id))
            seq_confidences.append(math.exp(logprob))

        seq_str = tokenizer.decode(seq_tokens)
        avg_conf = sum(seq_confidences) / \
            len(seq_confidences) if seq_confidences else 0.0
        sequences.append({"sequence": seq_str, "average_confidence": avg_conf})

    return {"sequences": sequences, "tokens": token_results}


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    model: LLMModelType = "qwen3-1.7b-4bit"
    logprobs = 5
    # prompt = "Write a 10 word short story."
    prompt = "The quick brown fox"
    response = generate(prompt, model, logprobs=logprobs,
                        verbose=True, max_tokens=20)

    results = read_logprobs(response, top_n=logprobs)
    save_file(results, f"{output_dir}/results.json")
