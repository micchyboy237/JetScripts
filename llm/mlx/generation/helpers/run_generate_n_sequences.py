import shutil
from unittest.mock import MagicMock
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.generation import chat, generate
from jet.llm.mlx.helpers.generate_n_sequences import generate_n_sequences
from jet.logger import logger
from jet.wordnet.sentence import count_sentences, split_sentences
import pytest
import os
from typing import Union, Dict, List, TypedDict
from jet.models.model_types import CompletionResponse, LLMModelType
from jet.llm.mlx.token_utils import get_tokenizer
from transformers import PreTrainedTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper
import math

MLXTokenizer = Union[TokenizerWrapper, PreTrainedTokenizer]


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    model: LLMModelType = "qwen3-1.7b-4bit"
    # prompt = "Write a 10 word short story."
    prompt = "The quick brown fox"
    n = 5
    sequences = generate_n_sequences(prompt, model, n)

    save_file({
        "prompt": prompt,
        "sequences": sequences
    }, f"{output_dir}/sequences.json")
