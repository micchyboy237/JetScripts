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


def extract_relation(
    question: str,
    model_path: LLMModelType = DEFAULT_MODEL,
    method: str = "stream_generate",
    max_tokens: int = 10,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> AnswerResult:
    """
    Extracts a relation from a given question using a language model.

    Args:
        question: The question to be answered.
        model_path: Path to the model (defaults to DEFAULT_MODEL).
        method: Generation method ("stream_generate" or "generate_step").
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.

    Returns:
        AnswerResult containing the relation, token ID, validity, method, and any error.
    """
    try:
        # Validate inputs
        validate_method(method)

        # Load model and tokenizer
        model_components = load_model_components(model_path)

        # Create and log prompt
        system_prompt = create_system_prompt(question)
        log_prompt_details(system_prompt, question, model_path)

        # Format messages and apply chat template
        messages = format_chat_messages(system_prompt, question)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Encode choices and setup generation parameters
        choice_token
