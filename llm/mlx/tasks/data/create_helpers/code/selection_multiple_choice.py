from typing import List, Dict, Optional, TypedDict
from uuid import uuid4
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


def load_model_components(model_path: LLMModelType) -> ModelComponents:
    """Loads model and tokenizer from the specified path."""
    try:
        model, tokenizer = load(resolve_model(model_path))
        return ModelComponents(model, tokenizer)
    except Exception as e:
        raise ModelLoadError(f"Error loading model or tokenizer: {e}")


def validate_method(method: str) -> None:
    """Validates the generation method."""
    valid_methods = ["stream_generate", "generate_step"]
    if method not in valid_methods:
        raise InvalidMethodError(
            f"Invalid method specified: {method}. Valid methods: {valid_methods}")


def create_system_prompt(choices: List[str]) -> str:
    """Creates a formatted system prompt with the given choices."""
    return f"Answer the following question by choosing one of the options provided without any additional text.\nOptions:\n{'\n'.join(choices)}"


def log_prompt_details(system_prompt: str, question: str, model_path: LLMModelType) -> None:
    """Logs system prompt, tokenized system prompt, and user question for debugging."""
    logger.gray("System:")
    logger.debug(system_prompt)
    logger.gray("Tokenized System:")
    logger.debug(tokenize_strings(system_prompt, model_path))
    logger.gray("User:")
    logger.debug(question)
    logger.newline()


def format_chat_messages(system_prompt: str, question: str) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]


def encode_choices(tokenizer: TokenizerWrapper, choices: List[str]) -> Dict[str, List[int]]:
    """Encodes each choice into tokens and logs the results."""
    choice_token_map = {}
    for choice in choices:
        tokens = tokenizer.encode(choice, add_special_tokens=False)
        choice_token_map[choice] = tokens
        logger.log(f"Tokens for '{choice}':",
                   tokens, colors=["GRAY", "ORANGE"])
    return choice_token_map


def setup_generation_parameters(
    tokenizer: TokenizerWrapper,
    choice_token_map: Dict[str, List[int]],
    temperature: float,
    top_p: float
) -> tuple:
    """Sets up logit bias, logits processors, sampler, and stop tokens for generation."""
    logit_bias = {tokens[0]: 0.0 for choice,
                  tokens in choice_token_map.items() if tokens}
    logits_processors = make_logits_processors(logit_bias=logit_bias)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    stop_tokens = tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
    return logits_processors, sampler, stop_tokens


def generate_answer_stream(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int],
    choices: List[str]
) -> tuple[str, int, List[int]]:
    """Generates an answer using stream_generate method."""
    answer = ""
    token_id = -1
    generated_tokens = []

    for output in stream_generate(
        model=model_components.model,
        tokenizer=model_components.tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        logits_processors=logits_processors,
        sampler=sampler
    ):
        if output.token in stop_tokens:
            break
        generated_tokens.append(output.token)
        token_id = output.token
        answer = model_components.tokenizer.decode(generated_tokens)
        if answer in choices:
            break

    return answer, token_id, generated_tokens


def generate_answer_step(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int],
    choices: List[str]
) -> tuple[str, int, List[int]]:
    """Generates an answer using generate_step method."""
    answer = ""
    token_id = -1
    generated_tokens = []

    input_ids = mx.array(model_components.tokenizer.encode(
        formatted_prompt, add_special_tokens=False))
    prompt_cache = None

    for token, _ in generate_step(
        model=model_components.model,
        prompt=input_ids,
        max_tokens=max_tokens,
        logits_processors=logits_processors,
        sampler=sampler,
        prompt_cache=prompt_cache
    ):
        if token in stop_tokens:
            break
        generated_tokens.append(token)
        token_id = token
        answer = model_components.tokenizer.decode(generated_tokens)
        if answer in choices:
            break

    return answer, token_id, generated_tokens


def validate_answer(answer: str, choices: List[str]) -> None:
    """Validates that the generated answer is one of the provided choices."""
    if answer not in choices:
        raise InvalidOutputError(
            f"Output '{answer}' is not one of the provided choices.")


def answer_multiple_choice(
    question: str,
    choices: List[str],
    model_path: LLMModelType = DEFAULT_MODEL,
    method: str = "stream_generate",
    max_tokens: int = 10,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> AnswerResult:
    """
    Answers a multiple-choice question using a language model.

    Args:
        question: The question to be answered.
        choices: List of possible answer choices.
        model_path: Path to the model (defaults to DEFAULT_MODEL).
        method: Generation method ("stream_generate" or "generate_step").
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.

    Returns:
        AnswerResult containing the answer, token ID, validity, method, and any error.
    """
    try:
        # Validate inputs
        validate_method(method)

        # Load model and tokenizer
        model_components = load_model_components(model_path)

        # Create and log prompt
        system_prompt = create_system_prompt(choices)
        log_prompt_details(system_prompt, question, model_path)

        # Format messages and apply chat template
        messages = format_chat_messages(system_prompt, question)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Encode choices and setup generation parameters
        choice_token_map = encode_choices(model_components.tokenizer, choices)
        logits_processors, sampler, stop_tokens = setup_generation_parameters(
            model_components.tokenizer, choice_token_map, temperature, top_p
        )

        # Generate answer based on method
        if method == "stream_generate":
            answer, token_id, _ = generate_answer_stream(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens, choices
            )
        else:
            answer, token_id, _ = generate_answer_step(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens, choices
            )

        # Validate the answer
        validate_answer(answer, choices)

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
