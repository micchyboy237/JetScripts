import json
import os
from jet.llm.mlx.client import MLXLMClient
from jet.logger import CustomLogger
from jet.transformers.formatters import format_json

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


model = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def get_models_example(client: MLXLMClient):
    """Example of using the .get_models method to list available models."""
    models = client.get_models()

    logger.debug("Available Models:")
    logger.success(json.dumps(models, indent=2))
    return models


def chatbot_example(client: MLXLMClient):
    """Example of using the .chat method for a conversational AI assistant."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's the capital of France?"},
    ]

    response = client.chat(
        messages=messages,
        model=model,
        max_tokens=100,
        temperature=0.7,
        stop=["\n\n"]
    )

    logger.debug("Chatbot Response:")
    logger.success(json.dumps(response, indent=2))
    return response


def streaming_chat_example(client: MLXLMClient):
    """Example of using the .stream_chat method for streaming chat completions."""
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
        stop=["\n\n"]
    ):
        if response["choices"]:
            content = response["choices"][0].get(
                "message", {}).get("content", "")
            full_response += content
            logger.success(content, flush=True)

            if response["choices"][0]["finish_reason"]:
                logger.newline()
                logger.orange(format_json(response))

    return full_response


def text_generation_example(client: MLXLMClient):
    """Example of using the .generate method for creative text generation."""
    prompt = "Once upon a time, in a distant kingdom, there lived a"

    response = client.generate(
        prompt=prompt,
        model=model,
        max_tokens=150,
        temperature=0.9,
        top_p=0.95,
        stop=["."],
    )

    logger.debug("Text Generation Response:")
    logger.success(json.dumps(response, indent=2))
    return response


def streaming_generate_example(client: MLXLMClient):
    """Example of using the .stream_generate method for streaming text generation."""
    prompt = "In a world where magic was real, the greatest wizard"

    logger.debug("Streaming Text Generation Response:")
    full_response = ""

    for response in client.stream_generate(
        prompt=prompt,
        model=model,
        max_tokens=150,
        temperature=0.9,
        top_p=0.95,
        stop=["."]
    ):
        if response["choices"]:
            content = response["choices"][0].get("text", "")
            full_response += content
            logger.success(content, flush=True)

            if response["choices"][0]["finish_reason"]:
                logger.newline()
                logger.orange(format_json(response))

    return full_response


def main():
    """Main function to run all examples."""
    # Initialize the client with default configuration
    client = MLXLMClient()

    logger.info("\n=== Get Models Example ===")
    get_models_example(client)

    logger.info("\n=== Chatbot Example ===")
    chatbot_example(client)

    logger.info("\n=== Streaming Chat Example ===")
    streaming_chat_example(client)

    logger.info("\n=== Text Generation Example ===")
    text_generation_example(client)

    logger.info("\n=== Streaming Text Generation Example ===")
    streaming_generate_example(client)


if __name__ == "__main__":
    main()
