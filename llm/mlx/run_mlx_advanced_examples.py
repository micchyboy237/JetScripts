import os
from jet.llm.mlx.client import MLXLMClient
from jet.logger import CustomLogger
from jet.transformers.formatters import format_json

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


def advanced_chat_with_logprobs_example(client: MLXLMClient):
    """Example using chat with logprobs to analyze token probabilities.
    Scenario: Generate a technical report summary, where logprobs=5 provides
    probability data for the top 5 token choices at each step.
    """
    messages = [
        {"role": "system", "content": "You are a technical writer for a research firm."},
        {"role": "user", "content": "Summarize a report on renewable energy advancements."},
    ]
    response = client.chat(
        messages=messages,
        logprobs=5  # Returns top 5 token probabilities for each generated token
    )
    logger.debug(
        "Advanced Chat with Logprobs Response (technical report summary):")
    logger.success(format_json(response))
    return response


def chat_with_role_mapping_example(client: MLXLMClient):
    """Example using chat with role_mapping to customize message formatting.
    Scenario: Format a customer support conversation with custom role prefixes.
    """
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
        messages=messages,
        role_mapping=role_mapping  # Customizes role prefixes in the prompt
    )
    logger.debug(
        "Chat with Role Mapping Response (customer support formatting):")
    logger.success(format_json(response))
    return response


def chat_with_logit_bias_example(client: MLXLMClient):
    """Example using logit_bias to favor specific tokens.
    Scenario: Generate a weather description favoring terms like 'sunny' or 'cloudy'.
    """
    messages = [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": "Describe the weather in New York."},
    ]
    # logit_bias increases probability of specific tokens (hypothetical IDs for "sunny", "cloudy")
    logit_bias = {32001: 2.0, 32002: 2.0}
    response = client.chat(
        messages=messages,
        logit_bias=logit_bias
    )
    logger.debug("Chat with Logit Bias Response (favoring 'sunny', 'cloudy'):")
    logger.success(format_json(response))
    return response


def chat_with_tools_example(client: MLXLMClient):
    """Example using tools for structured function calls.
    Scenario: Structure a weather query for a weather API call.
    """
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
        messages=messages,
        tools=tools
    )
    logger.debug("Chat with Tools Response (structured for weather API):")
    logger.success(format_json(response))
    return response


def streaming_chat_with_stop_example(client: MLXLMClient):
    """Example using streaming with stop tokens to terminate output.
    Scenario: Generate a smartphone product description, stopping at specific phrases.
    """
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


def streaming_chat_with_temperature_example(client: MLXLMClient):
    """Example using streaming with temperature for creative output.
    Scenario: Generate a creative tagline for a travel agency.
    """
    messages = [
        {"role": "system", "content": "You are a creative copywriter for a travel agency."},
        {"role": "user", "content": "Suggest a catchy tagline for our global travel packages."},
    ]
    responses = client.stream_chat(
        messages=messages,
        temperature=1.0  # Higher temperature for more creative output
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


def text_generation_with_logprobs_example(client: MLXLMClient):
    """Example using logprobs to analyze token probabilities.
    Scenario: Generate a product name for a smartwatch.
    """
    prompt = "Suggest a name for a new smartwatch: "
    response = client.generate(
        prompt=prompt,
        logprobs=3  # Returns top 3 token probabilities for each generated token
    )
    logger.debug(
        "Text Generation with Logprobs (smartwatch name with token probabilities):")
    logger.success(format_json(response))
    return response


def chat_with_repetition_penalty_example(client: MLXLMClient):
    """Example using repetition penalty to reduce redundant phrases.
    Scenario: Write a brief company mission statement.
    """
    messages = [
        {"role": "system", "content": "You are a corporate communications specialist."},
        {"role": "user", "content": "Draft a mission statement for our tech startup."},
    ]
    response = client.chat(
        messages=messages,
        repetition_penalty=1.2,  # Penalizes repeated tokens to reduce redundancy
        repetition_context_size=30  # Considers last 30 tokens for repetition check
    )
    logger.debug(
        "Chat with Repetition Penalty (non-repetitive mission statement):")
    logger.success(format_json(response))
    return response


def text_generation_with_xtc_example(client: MLXLMClient):
    """Example using XTC for concise text generation.
    Scenario: Summarize a news article with token compression.
    """
    prompt = "Summarize: A new electric car was unveiled with a 400-mile range and advanced AI."
    response = client.generate(
        prompt=prompt,
        xtc_probability=0.5,  # Apply XTC more often
        xtc_threshold=0.4     # Only top tokens with probs > 0.4
    )
    logger.debug("Text Generation with XTC (concise news summary):")
    logger.success(format_json(response))
    return response


def main():
    """Main function to run all advanced examples."""
    client = MLXLMClient()
    logger.info("\n=== Advanced Chat with Logprobs Example ===")
    advanced_chat_with_logprobs_example(client)
    logger.info("\n=== Chat with Role Mapping Example ===")
    chat_with_role_mapping_example(client)
    logger.info("\n=== Chat with Logit Bias Example ===")
    chat_with_logit_bias_example(client)
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


if __name__ == "__main__":
    main()
