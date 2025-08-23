import os
from typing import Any, Dict
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import LLMModelKey
from jet.transformers.formatters import format_json

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


def encode_decode_example(client: MLX) -> Dict[str, Any]:
    """Encode and decode single and multiple text inputs using the model's tokenizer.

    Args:
        client: An object with a tokenizer attribute (instance of TokenizerWrapper).

    Returns:
        Dict containing the input texts, encoded tokens, decoded texts, and their types.
    """
    single_text = "Hello, how are you today?"
    multiple_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries."
    ]

    # Encode single text
    single_encoded = client.tokenizer.encode(single_text, return_tensors=None)
    single_decoded = client.tokenizer.decode(single_encoded)

    # Encode multiple texts with proper padding and truncation
    multiple_encoded = client.tokenizer.encode(
        multiple_texts, return_tensors=None, padding=True, truncation=True
    )
    multiple_decoded = client.tokenizer.batch_decode(multiple_encoded)

    # Prepare results for logging
    result = {
        "single_text": {
            "input": single_text,
            "encoded": single_encoded,
            "decoded": single_decoded,
            "encoded_type": type(single_encoded).__name__,
            "decoded_type": type(single_decoded).__name__
        },
        "multiple_texts": {
            "inputs": multiple_texts,
            "encoded": multiple_encoded,
            "decoded": multiple_decoded,
            "encoded_type": type(multiple_encoded).__name__,
            "decoded_type": type(multiple_decoded).__name__
        }
    }

    logger.debug("Encode/Decode Example (single and multiple text inputs):")
    logger.success(format_json(result))
    return result


def advanced_chat_with_logprobs_example(client: MLX):
    messages = [
        {"role": "system", "content": "You are a technical writer for a research firm."},
        {"role": "user", "content": "Summarize a report on renewable energy advancements."},
    ]
    response = client.chat(
        verbose=True,
        messages=messages,
        logprobs=5
    )
    logger.debug(
        "Advanced Chat with Logprobs Response (technical report summary):")
    logger.success(format_json(response))
    return response


def chat_with_role_mapping_example(client: MLX):
    messages = [
        {"role": "system", "content": "You are a customer support assistant for a tech company."},
        {"role": "user", "content": "My laptop won't turn on. What should I do?"},
    ]
    role_mapping = {
        "system": "Support Agent Brief: ",
        "user": "Customer Inquiry: ",
        "assistant": "Support Response: "
    }
    response = client.chat(
        verbose=True,
        messages=messages,
        role_mapping=role_mapping
    )
    logger.debug(
        "Chat with Role Mapping Response (customer support formatting):")
    logger.success(format_json(response))
    return response


def chat_with_logit_bias_example(client: MLX) -> Dict[str, Any]:
    """Example using logit_bias to favor specific tokens.
    Scenario: Generate a weather description favoring terms like 'sunny' or 'cloudy'.

    Args:
        client: An object with a tokenizer and chat method (instance of MLX).

    Returns:
        Dict containing the chat response.
    """
    messages = [
        {"role": "system", "content": "You are a weather assistant. Provide a general overview of New York's weather by season, avoiding speculative current weather."},
        {"role": "user", "content": "Describe the weather in New York."},
    ]
    # Encode the words "sunny" and "cloudy" to get their token IDs
    sunny_tokens = client.tokenizer.encode(
        "sunny", return_tensors=None, add_special_tokens=False)
    cloudy_tokens = client.tokenizer.encode(
        "cloudy", return_tensors=None, add_special_tokens=False)

    logger.debug(f"Sunny tokens: {sunny_tokens}")
    logger.debug(f"Cloudy tokens: {cloudy_tokens}")

    # Create logit_bias dictionary, including all non-special token IDs for "sunny" and "cloudy"
    logit_bias = {}
    for token_id in sunny_tokens:
        logit_bias[token_id] = 5.0
    for token_id in cloudy_tokens:
        logit_bias[token_id] = 5.0

    if not logit_bias:
        logger.warning(
            "No valid non-special token IDs found for 'sunny' or 'cloudy'. Proceeding without logit bias.")
        response = client.chat(
            verbose=True, messages=messages)
    else:
        response = client.chat(
            verbose=True,
            messages=messages,
            logit_bias=logit_bias,
            temperature=0.0
        )

    logger.debug("Chat with Logit Bias Response (favoring 'sunny', 'cloudy'):")
    logger.success(format_json(response))
    return response


def chat_with_tools_example(client: MLX):
    messages = [
        {"role": "system", "content": "You are a weather assistant with access to a weather API."},
        {"role": "user", "content": "What's the weather like in New York?"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    response = client.chat(
        verbose=True,
        messages=messages,
        tools=tools
    )
    logger.debug("Chat with Tools Response (structured for weather API):")
    logger.success(format_json(response))
    return response


def streaming_chat_with_stop_example(client: MLX):
    messages = [
        {"role": "system", "content": "You are a marketing assistant for tech products."},
        {"role": "user", "content": "Write a brief description for a new smartphone."},
    ]
    responses = client.stream_chat(
        messages=messages,
        stop=["\n\n", "Features:", "Specifications:"]
    )
    logger.debug(
        "Streaming Chat with Stop Tokens (stops at '\n\n', 'Features:' or 'Specifications:'):")
    full_response = ""
    for response in responses:
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content", "")
            full_response += content
            logger.success(content, flush=True)
            if choice["finish_reason"]:
                logger.newline()
                logger.orange(format_json(response))
    return full_response


def streaming_chat_with_temperature_example(client: MLX):
    messages = [
        {"role": "system", "content": "You are a creative copywriter for a travel agency."},
        {"role": "user", "content": "Suggest a catchy tagline for our global travel packages."},
    ]
    responses = client.stream_chat(
        messages=messages,
        temperature=1.0
    )
    logger.debug("Streaming Chat with High Temperature (creative tagline):")
    full_response = ""
    for response in responses:
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content", "")
            full_response += content
            if choice["finish_reason"]:
                logger.newline()
                logger.orange(format_json(response))
    logger.newline()
    return full_response


def text_generation_with_logprobs_example(client: MLX):
    prompt = "Suggest a name for a new smartwatch: "
    response = client.generate(
        prompt=prompt,
        logprobs=3
    )
    logger.debug(
        "Text Generation with Logprobs (smartwatch name with token probabilities):")
    logger.success(format_json(response))
    return response


def chat_with_repetition_penalty_example(client: MLX):
    messages = [
        {"role": "system", "content": "You are a corporate communications specialist."},
        {"role": "user", "content": "Draft a mission statement for our tech startup."},
    ]
    response = client.chat(
        verbose=True,
        messages=messages,
        repetition_penalty=1.2,
        repetition_context_size=30
    )
    logger.debug(
        "Chat with Repetition Penalty (non-repetitive mission statement):")
    logger.success(format_json(response))
    return response


def text_generation_with_xtc_example(client: MLX):
    prompt = "Summarize: A new electric car was unveiled with a 400-mile range and advanced AI."
    response = client.generate(
        prompt=prompt,
        xtc_probability=0.5,
        xtc_threshold=0.4
    )
    logger.debug("Text Generation with XTC (concise news summary):")
    logger.success(format_json(response))
    return response


if __name__ == "__main__":
    """Main function to run all advanced examples."""
    model: LLMModelKey = "qwen3-1.7b-4bit"
    client = MLXModelRegistry.load_model(model)

    logger.info("\n=== Encode/Decode Example ===")
    encode_decode_example(client)
    logger.info("\n=== Chat with Logit Bias Example ===")
    chat_with_logit_bias_example(client)
    logger.info("\n=== Advanced Chat with Logprobs Example ===")
    advanced_chat_with_logprobs_example(client)
    logger.info("\n=== Chat with Role Mapping Example ===")
    chat_with_role_mapping_example(client)
    logger.info("\n=== Chat with Tools Example ===")
    chat_with_tools_example(client)
    logger.info("\n=== Streaming Chat with Stop Tokens Example ===")
    streaming_chat_with_stop_example(client)
    logger.info("\n=== Streaming Chat with Temperature Example ===")
    streaming_chat_with_temperature_example(client)
    logger.info("\n=== Text Generation with Logprobs Example ===")
    text_generation_with_logprobs_example(client)
    logger.info("\n=== Chat with Repetition Penalty Example ===")
    chat_with_repetition_penalty_example(client)
    logger.info("\n=== Text Generation with XTC Example ===")
    text_generation_with_xtc_example(client)
