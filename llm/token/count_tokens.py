from jet.llm.token import token_counter
from jet.logger import logger


def main():
    # Test case 1: Count tokens in a simple text
    text = "This is a simple test."
    tokens_text = token_counter(text=text)
    logger.log("Number of tokens for text:", {
               tokens_text}, colors=["GRAY", "SUCCESS"])

    # Test case 2: Count tokens from a list of messages
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you! How can I assist?"},
    ]
    tokens_messages = token_counter(messages=messages)
    logger.log("Number of tokens for messages:", {
               tokens_messages}, colors=["GRAY", "SUCCESS"])

    # Test case 3: Count tokens with a tool call
    messages_with_tool = [
        {
            "role": "user",
            "content": "What is the weather?",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": "city=London&date=today",
                    }
                }
            ],
        }
    ]
    tokens_tool = token_counter(messages=messages_with_tool)
    logger.log("Number of tokens for messages with tool calls:", {
               tokens_tool}, colors=["GRAY", "SUCCESS"])

    # Test case 4: Count tokens with images
    messages_with_images = [
        {
            "role": "user",
            "content": "Describe this image:",
            "content": [
                {"type": "image_url", "image_url": "http://example.com/image1.jpg"},
                {"type": "text", "text": "A beautiful sunset."},
            ],
        }
    ]
    tokens_images = token_counter(messages=messages_with_images)
    logger.log("Number of tokens for messages with images:", {
               tokens_images}, colors=["GRAY", "SUCCESS"])

    # Test case 5: Count tokens with empty inputs
    try:
        tokens_empty = token_counter()
    except ValueError as e:
        logger.success(f"Error for empty inputs: {e}")


if __name__ == "__main__":
    main()
