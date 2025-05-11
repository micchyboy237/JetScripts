from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.config import DEFAULT_MODEL
from jet.llm.mlx.mlx_types import ModelType
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


def summarize_text(text: str, max_tokens: int = 10) -> AnswerResult:
    """
    Summarizes a given text.

    Args:
        text: The text to be summarized.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        AnswerResult containing the summary, token ID, validity, method, and any error.
    """
    try:
        # Load model and tokenizer
        model_components = load_model_components(resolve_model("model_path"))

        # Create and log prompt
        system_prompt = create_system_prompt(["Text to be summarized"])
        log_prompt_details(system_prompt, text, model_path)

        # Format messages and apply chat template
        messages = format_chat_messages(system_prompt, text)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Encode choices and setup generation parameters
        choice_token_map = encode_choices(model_components.tokenizer, ["Text to be summarized"])
        logits_processors, sampler, stop_tokens = setup_generation_parameters(
            model_components.tokenizer, choice_token_map, 0.0, 0.9
        )

        # Generate answer based on method
        if model_components.tokenizer.model_type == ModelType.STREAM_GENERATION:
            answer, token_id, _ = generate_answer_stream(
                model_components,