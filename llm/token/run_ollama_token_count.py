from jet.llm.ollama import OLLAMA_HF_MODELS, count_tokens, get_token_max_length
from jet.llm.llm_types import Message
from jet.logger import logger

ollama_hf_models = {
    "llama3.1": OLLAMA_HF_MODELS["llama3.1"],
    "mistral": OLLAMA_HF_MODELS["mistral"],
}


def main_count_tokens():
    # Example text
    text = "This is a sample text for token counting."

    # Iterate over the models and count tokens for each
    for model_key, model_name in ollama_hf_models.items():
        logger.newline()
        logger.log("Counting tokens for", model_key,
                   "...", colors=["WHITE", "DEBUG"])
        token_count = count_tokens(model_name, text)
        logger.debug(f"{model_key} count:")
        logger.success(token_count)


def main_get_token_max_length():
    for model_key in ollama_hf_models.keys():
        logger.newline()
        logger.log("Getting token max length for", model_key,
                   "...", colors=["WHITE", "DEBUG"])
        token_max_length = get_token_max_length(model_key)
        logger.debug(f"{model_key} token max length:")
        logger.success(token_max_length)


def main_count_tokens_batch():
    texts = [
        "This is the first example sentence.",
        "Here is another text to test the function.",
        "Let's see how the token count works with multiple items in the batch.",
        "This is the last sentence in the example batch."
    ]

    # Iterate over the models and count tokens for this batch
    for model_key, model_name in ollama_hf_models.items():
        logger.newline()
        logger.log("Counting tokens batch for", model_key,
                   "...", colors=["WHITE", "DEBUG"])
        token_count = count_tokens(model_name, texts)
        logger.debug(f"{model_key} count:")
        logger.success(token_count)


def main_count_tokens_chat_simple_user_assistant():
    messages = [
        {"role": "system", "content": "This is the system prompt for the conversation."},
        {"role": "user", "content": "Hello! How can I create a Python dictionary?"},
        {"role": "assistant", "content": "You can create a Python dictionary using curly braces, like this: `my_dict = {'key': 'value'}`."},
        {"role": "user", "content": "Thank you! Can you explain how to access values?"},
        {"role": "assistant",
            "content": "Sure! You can access values by using the key, like `my_dict['key']`."}
    ]

    for model_key, model_name in ollama_hf_models.items():
        logger.newline()
        logger.log(f"Counting tokens for simple user-assistant messages using {
                   model_key}...", colors=["WHITE", "DEBUG"])
        token_count = count_tokens(model_name, messages)
        logger.debug(f"{model_key} token count:")
        logger.success(token_count)


def main_count_tokens_chat_tool_interaction():
    messages: list[Message] = [
        {
            "role": "user",
            "content": "What is the mystery function on 5 and 6?"
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "mystery",
                        "arguments": {
                            "a": 5,
                            "b": 6
                        }
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": -11
        }
    ]

    for model_key, model_name in ollama_hf_models.items():
        logger.newline()
        logger.log(f"Counting tokens for tool interaction messages using {
                   model_key}...", colors=["WHITE", "DEBUG"])
        token_count = count_tokens(model_name, messages)
        logger.debug(f"{model_key} token count:")
        logger.success(token_count)


if __name__ == "__main__":
    logger.newline()
    logger.info("main_get_token_max_length()...")
    main_get_token_max_length()

    logger.newline()
    logger.info("main_count_tokens()...")
    main_count_tokens()

    logger.newline()
    logger.info("main_count_tokens_batch()...")
    main_count_tokens_batch()

    logger.newline()
    logger.info("main_count_tokens_chat_simple_user_assistant()...")
    main_count_tokens_chat_simple_user_assistant()

    logger.newline()
    logger.info("main_count_tokens_chat_tool_interaction()...")
    main_count_tokens_chat_tool_interaction()
