from typing import List, Dict
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


def segment_text(text: str, max_length: int = 1000) -> ChatMessage:
    """
    Segments the input text into smaller chunks.

    Args:
        text: The input text to be segmented.
        max_length: The maximum length of each chunk.

    Returns:
        ChatMessage containing the segmented text.
    """
    # Split the text into sentences
    sentences = tokenize_strings(text, max_length)

    # Segment each sentence into words
    words = []
    for sentence in sentences:
        words.extend(tokenize_strings(sentence, max_length))

    # Join the words back into a text
    text = ' '.join(words)

    return ChatMessage(text, text)


def generate_answer_stream(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int = 1000,
    logits_processors: list,
    sampler,
    stop_tokens: List[int],
    choices: List[str]
) -> tuple[str, int, List[int]]:
    """
    Generates an answer using the provided inputs.

    Args:
        model_components: The model and tokenizer components.
        formatted_prompt: The formatted prompt.
        max_tokens: The maximum number of tokens to generate.
        logits_processors: The logits processor.
        sampler: The sampler.
        stop_tokens: The stop tokens.
        choices: The possible answer choices.

    Returns:
        tuple[str, int
