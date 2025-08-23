# jet_python_modules/jet/llm/mlx/usage_examples.py

import os
import shutil
from typing import Iterator, Union, Literal
from pydantic.json_schema import JsonSchemaValue
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import Message, CompletionResponse
from jet.logger import CustomLogger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import LLMModelKey
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


def streaming_chat_with_response_format_json(client: MLX) -> Iterator[CompletionResponse]:
    """
    Demonstrate streaming chat with JSON response format.
    Requests a JSON response for a user query about a book summary.
    """
    # Define messages for the chat
    messages: list[Message] = [
        {"role": "user", "content": "Provide a summary of '1984' by George Orwell in JSON format."}
    ]

    # Stream chat with JSON response format
    response = ""
    for response_stream in client.stream_chat(
        messages=messages,
        response_format="json",
        verbose=True,
        log_dir=f"{OUTPUT_DIR}/chat_json"
    ):
        if response_stream.get("choices"):
            content = response_stream["choices"][0].get(
                "message", {}).get("content", "")
            response += content

    logger.debug("Streaming chat with response_format JSON result:")
    logger.success(format_json(response))


def streaming_chat_with_response_format_json_schema(client: MLX) -> Iterator[CompletionResponse]:
    """
    Demonstrate streaming chat with JSON schema response format.
    Requests a structured JSON response for a user query about a book summary.
    """
    # Define a JSON schema for the book summary response
    book_summary_schema: JsonSchemaValue = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "summary": {"type": "string"},
            "year": {"type": "integer"}
        },
        "required": ["title", "author", "summary", "year"]
    }

    # Define messages for the chat
    messages: list[Message] = [
        {"role": "user", "content": "Provide a summary of '1984' by George Orwell in JSON format."}
    ]

    # Stream chat with JSON schema response format
    response = ""
    for response_stream in client.stream_chat(
        messages=messages,
        response_format=book_summary_schema,
        verbose=True,
        log_dir=f"{OUTPUT_DIR}/chat_json_schema"
    ):
        if response_stream.get("choices"):
            content = response_stream["choices"][0].get(
                "message", {}).get("content", "")
            response += content

    logger.debug("Streaming chat with response_format JSON schema result:")
    logger.success(format_json(response))


def streaming_generate_with_response_format_json(client: MLX) -> Iterator[CompletionResponse]:
    """
    Demonstrate streaming generate with JSON response format.
    Requests a JSON response for a prompt about a weather forecast.
    """
    # Define the prompt
    prompt: str = "Provide today's weather forecast for London in JSON format."

    # Stream generate with JSON response format
    response = ""
    for response_stream in client.stream_generate(
        prompt=prompt,
        response_format="json",
        verbose=True,
        log_dir=f"{OUTPUT_DIR}/generate_json"
    ):
        if response_stream.get("choices"):
            content = response_stream["choices"][0].get("text", "")
            response += content

    logger.debug("Streaming generate with response_format JSON result:")
    logger.success(format_json(response))


def streaming_generate_with_response_format_json_schema(client: MLX) -> Iterator[CompletionResponse]:
    """
    Demonstrate streaming generate with JSON schema response format.
    Requests a structured JSON response for a prompt about a weather forecast.
    """
    # Define a JSON schema for the weather forecast response
    weather_forecast_schema: JsonSchemaValue = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temperature": {"type": "number"},
            "condition": {"type": "string"},
            "date": {"type": "string", "format": "date"}
        },
        "required": ["city", "temperature", "condition", "date"]
    }

    # Define the prompt
    prompt: str = "Provide today's weather forecast for London in JSON format."

    # Stream generate with JSON schema response format
    response = ""
    for response_stream in client.stream_generate(
        prompt=prompt,
        response_format=weather_forecast_schema,
        verbose=True,
        log_dir=f"{OUTPUT_DIR}/generate_json_schema"
    ):
        if response_stream.get("choices"):
            content = response_stream["choices"][0].get("text", "")
            response += content

    logger.debug("Streaming generate with response_format JSON schema result:")
    logger.success(format_json(response))


if __name__ == "__main__":
    """Main function to run all advanced examples."""
    model: LLMModelKey = "qwen3-1.7b-4bit"
    client = MLXModelRegistry.load_model(model)

    logger.info("\n=== Streaming chat with response_format JSON example ===")
    client.reset_model()
    streaming_chat_with_response_format_json(client)

    logger.info(
        "\n=== Streaming chat with response_format JSON schema example ===")
    client.reset_model()
    streaming_chat_with_response_format_json_schema(client)

    logger.info("\n=== Streaming generate with response_format JSON example ===")
    client.reset_model()
    streaming_generate_with_response_format_json(client)

    logger.info(
        "\n=== Streaming generate with response_format JSON schema example ===")
    client.reset_model()
    streaming_generate_with_response_format_json_schema(client)
