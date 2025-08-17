from typing import List, Dict, Optional
from jet.llm.mlx.config import DEFAULT_MODEL
from jet.models.model_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.token_utils import tokenize_strings
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate, generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import TokenizerWrapper

# Custom exceptions for specific error cases


class ModelLoadError(Exception):
    pass


class InvalidMethodError(Exception):
    pass


class InvalidOutputError(Exception):
    pass

# Type definitions for structured data


class ChatMessage(TypedDict):
    role: str
    content: str


class AnswerResult(TypedDict):
    answer: str
    token_id: int
    is_valid: bool
    method: str
    error: Optional[str]


def answer_yes_no(
    question: str,
    model_path: LLMModelType = DEFAULT_MODEL,
    method: str = "stream_generate",
    max_tokens: int = 1,
    temperature: float = 0.1,
    top_p: float = 0.1
) -> AnswerResult:
    model_path = resolve_model(model_path)

    try:
        try:
            model, tokenizer = load(resolve_model(model_path))
        except Exception as e:
            raise ModelLoadError(f"Error loading model or tokenizer: {e}")

        try:
            model, tokenizer = load(resolve_model(model_path))
        except Exception as e:
            raise ModelLoadError(f"Error loading model or tokenizer: {e}")

        if method not in ["stream_generate", "generate_step"]:
            raise InvalidMethodError(
                f"Invalid method specified: {method}. Valid methods: {['stream_generate', 'generate_step']}"
            )

        messages: List[ChatMessage] = [
            {"role": "system", "content": "Answer the following question with only 'Yes' or 'No'. Ensure accuracy."},
            {"role": "user", "content": question}
        ]

        try:
            formatted_prompt: str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            raise PromptFormattingError(f
