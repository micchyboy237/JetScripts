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


def yes_no_question(question: str) -> AnswerResult:
    """
    Generates a yes/no question using a language model.

    Args:
        question: The question to be answered.

    Returns:
        AnswerResult containing the answer, token ID, validity, method, and any error.
    """
    # Create a yes/no system prompt
    system_prompt = create_system_prompt(["Yes", "No"])

    # Log the system prompt
    log_prompt_details(system_prompt, question, "DEFAULT_MODEL")

    # Format the message
    messages = format_chat_messages(system_prompt, question)

    # Generate the answer
    answer = generate_answer_stream(
        resolve_model("DEFAULT_MODEL"), messages, 10, 0.0, 0.9
    )

    # Validate the answer
    validate_answer(answer, ["Yes", "No"])

    return AnswerResult(answer=answer, token_id=-1, is_valid=True, method="stream_generate", error=None)


def yes_no_prompt(prompt: str) -> str:
    """
    Generates a yes/no prompt using a language model.

    Args:
        prompt: The prompt to be generated.

    Returns:
        str: The generated prompt.
    """
    # Create a yes/no system prompt
    system_prompt = create_system_prompt(["Yes", "No"])

    # Log the system prompt
    log_prompt_details(system_prompt, prompt, "DEFAULT_MODEL")

    # Format the message
    messages = format_chat_messages(system_prompt, prompt)

    # Generate the answer
    answer = generate_answer_stream(
        resolve_model("DEFAULT_MODEL"), messages, 10, 0.0, 0.9
    )

    # Validate the answer
    validate_answer(answer, ["Yes", "No"])

    return AnswerResult(answer=answer, token_id=-1, is_valid=True, method="stream_generate", error=None)


def yes_no_game(prompt: str, max_tokens: int = 10) -> AnswerResult:
    """
    Generates a yes/no game using a language model.

    Args:
        prompt: The prompt to be generated.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        AnswerResult containing the answer, token ID, validity, method, and any error.
    """
    # Create a yes/no system prompt
    system_prompt = create_system_prompt(["Yes", "No"])

    # Log the system prompt
    log_prompt_details(system_prompt, prompt, "DEFAULT_MODEL")

    # Format the message
    messages = format_chat_messages(system_prompt, prompt)

    # Generate the answer
    answer = generate_answer_stream(
        resolve_model("DEFAULT_MODEL"), messages, max_tokens, 0.0, 0.9
    )

    # Validate the answer
    validate_answer(answer, ["Yes", "No"])

    return AnswerResult(answer=answer, token_id=-1, is_valid=True, method="stream_generate", error=None)


def yes_no_game_loop(prompt: str, max_tokens: int = 10) -> AnswerResult:
    """
    Generates a yes/no game loop using a language model.

    Args:
        prompt: The prompt to be generated.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        AnswerResult containing the answer, token ID, validity, method, and any error.
    """
    # Generate the yes/no question
    answer = yes_no_question(prompt)

    # Validate the answer
    validate_answer(answer, ["Yes", "No"])

    # Generate the yes/no game loop
    answer = generate_answer_stream(
        resolve_model("DEFAULT_MODEL"), [answer], max_tokens, 0.0, 0.9
    )

    # Validate the answer
    validate_answer(answer, ["Yes", "No"])

    return AnswerResult(answer=answer, token_id=-1, is_valid=True, method="stream_generate", error=None)


def main():
    # Generate a yes/no question
    print("Yes/No Question:")
    print(yes_no_question("Is the sun shining today?"))

    # Generate a yes/no game loop
    print("\nYes/No Game Loop:")
    print(yes_no_game_loop("Is the moon full tonight?"))

    # Generate a yes/no game
    print("\nYes/No Game:")
    print(yes_no_game("Is the weather nice today?"))

    # Validate the answers
    print("\nValidating Answers:")
    print(validate_answer(yes_no_question(
        "Is the sun shining today?"), ["Yes", "No"]))
    print(validate_answer(yes_no_game_loop(
        "Is the moon full tonight?"), ["Yes", "No"]))
    print(validate_answer(yes_no_game(
        "Is the weather nice today?"), ["Yes", "No"]))


if __name__ == "__main__":
    main()
