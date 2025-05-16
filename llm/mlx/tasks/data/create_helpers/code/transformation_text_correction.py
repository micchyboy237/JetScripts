from typing import List, Dict, Optional, TypedDict
from uuid import uuid4
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


def correct_text(text: str, model_path: ModelType = DEFAULT_MODEL, method: str = "stream_generate", max_tokens: int = 10, temperature: float = 0.0, top_p: float = 0.9) -> AnswerResult:
    """
    Corrects the text using a language model.

    Args:
        text: The text to be corrected.
        model_path: Path to the model (defaults to DEFAULT_MODEL).
        method: Generation method ("stream_generate" or "generate_step").
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.

    Returns:
        AnswerResult containing the corrected answer, token ID, validity, method, and any error.
    """
    try:
        # Validate inputs
        validate_method(method)

        # Load model and tokenizer
        model_components = load_model_components(model_path)

        # Create and log prompt
        system_prompt = create_system_prompt(["Correct this sentence."]))
        log_prompt_details(system_prompt, text, model_path)

        # Format messages and apply chat template
        messages = format_chat_messages(system_prompt, text)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Encode choices and setup generation parameters
        choice_token_map = encode_choices(model_components.tokenizer, ["Correct this sentence."])
        logits_processors, sampler, stop_tokens = setup_generation_parameters(
            model_components.tokenizer, choice_token_map, temperature, top_p
        )

        # Generate answer based on method
        if method == "stream_generate":
            answer, token_id, _ = generate_answer_stream(
                model_components, formatted_prompt, max_tokens, logits_processors, sampler, stop_tokens, ["Correct this sentence."])
        else:
            answer, token_id, _ = generate_answer_step(
                model_components, formatted_prompt, max_tokens, logits_processors, sampler, stop_tokens, ["Correct this sentence."])
        answer = model_components.tokenizer.decode(answer)
        if answer in ["Correct this sentence.", "Correct this sentence."]:
            break

        # Validate the answer
        validate_answer(answer, ["Correct this sentence.", "Correct this sentence."])

        return AnswerResult(
            answer=answer,
            token_id=token_id,
            is_valid=True,
            method=method,
            error=None
        )

    except Exception as e:
        return AnswerResult(
            answer="",
            token_id=-1,
            is_valid=False,
            method=method,
            error=str(e)
        )