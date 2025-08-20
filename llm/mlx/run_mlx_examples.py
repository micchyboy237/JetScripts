import json
import os
from jet.llm.mlx.base import MLX
from jet.models.model_types import LLMModelKey, LLMModelType
from jet.logger import CustomLogger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.transformers.formatters import format_json

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)


model: LLMModelKey = "qwen3-1.7b-4bit"

MLX_LOG_DIR = f"{script_dir}/generated/run_mlx_examples"


def get_models_example(model_id: LLMModelType):
    """Example of using the .get_models method to list available models."""
    client = MLXModelRegistry.load_model(model_id)
    models = client.get_models()

    logger.debug("Available Models:")
    logger.success(json.dumps(models, indent=2))
    return models


def chatbot_example(model_id: LLMModelType):
    """Example of using the .chat method for a conversational AI assistant."""
    client = MLXModelRegistry.load_model(model_id)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's the capital of France?"},
    ]

    response = client.chat(
        messages=messages,
        model=model,
        max_tokens=100,
        temperature=0.7,
        stop=["\n\n"],
        log_dir=MLX_LOG_DIR
    )

    logger.debug("Chatbot Response:")
    logger.success(json.dumps(response, indent=2))
    return response


def streaming_chat_example(model_id: LLMModelType):
    """Example of using the .stream_chat method for streaming chat completions."""
    client = MLXModelRegistry.load_model(model_id)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Tell me a short story about a brave knight."},
    ]

    logger.debug("Streaming Chat Response:")
    full_response = ""

    for response in client.stream_chat(
        messages=messages,
        model=model,
        max_tokens=200,
        temperature=0.8,
        stop=["\n\n"],
        log_dir=MLX_LOG_DIR
    ):
        if response["choices"]:
            content = response["choices"][0].get(
                "message", {}).get("content", "")
            full_response += content
            logger.success(content, flush=True)

            if response["choices"][0]["finish_reason"]:
                logger.newline()

    return full_response


def text_generation_example(model_id: LLMModelType):
    """Example of using the .generate method for creative text generation."""
    client = MLXModelRegistry.load_model(model_id)
    prompt = "Once upon a time, in a distant kingdom, there lived a"

    response = client.generate(
        prompt=prompt,
        model=model,
        max_tokens=150,
        temperature=0.9,
        top_p=0.95,
        stop=["."],
        log_dir=MLX_LOG_DIR
    )

    logger.debug("Text Generation Response:")
    logger.success(json.dumps(response, indent=2))
    return response


def streaming_generate_example(model_id: LLMModelType):
    """Example of using the .stream_generate method for streaming text generation."""
    client = MLXModelRegistry.load_model(model_id)
    prompt = "In a world where magic was real, the greatest wizard"

    logger.debug("Streaming Text Generation Response:")
    full_response = ""

    for response in client.stream_generate(
        prompt=prompt,
        model=model,
        max_tokens=150,
        temperature=0.9,
        top_p=0.95,
        stop=["."],
        log_dir=MLX_LOG_DIR
    ):
        if response["choices"]:
            content = response["choices"][0].get("text", "")
            full_response += content
            logger.success(content, flush=True)

            if response["choices"][0]["finish_reason"]:
                logger.newline()

    return full_response


def main():
    """Main function to run all examples."""
    model_id: LLMModelType = "qwen3-1.7b-4bit"

    logger.info("\n=== Get Models Example ===")
    get_models_example(model_id)

    logger.info("\n=== Chatbot Example ===")
    chatbot_example(model_id)

    logger.info("\n=== Streaming Chat Example ===")
    streaming_chat_example(model_id)

    logger.info("\n=== Text Generation Example ===")
    text_generation_example(model_id)

    logger.info("\n=== Streaming Text Generation Example ===")
    streaming_generate_example(model_id)


if __name__ == "__main__":
    main()
    logger.orange(f"Logs: {log_file}")
