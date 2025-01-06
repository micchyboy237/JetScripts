import json
from jet.llm.token import token_counter, filter_texts
from jet.logger import logger
from llama_index.core.base.llms.types import ChatMessage, MessageRole

sample_texts: list[str] = [
    "This is the first example sentence.",
    "Here is another text to test the function.",
    "Let's see how the token count works with multiple items in the batch.",
    "This is the last sentence in the example batch."
]
sample_text = "\n".join(sample_texts)
sample_chat_messages: list[ChatMessage] = [
    ChatMessage(
        role=MessageRole.SYSTEM if i % 2 == 0 else MessageRole.USER,
        content=text
    )
    for i, text in enumerate(sample_texts)
]

if __name__ == "__main__":
    models = ["llama3.1"]
    ollama_models = {}

    logger.info("Texts:")
    logger.debug(json.dumps(sample_texts, indent=2))

    logger.info("Count tokens for: str")
    for model_name in models:
        count = token_counter(sample_text, model_name)
        logger.log("Tokens:", count, colors=["DEBUG", "SUCCESS"])

    logger.info("Count batch tokens for: list[str]")
    for model_name in models:
        count = token_counter(sample_texts, model_name)
        logger.log("Tokens:", count, colors=["DEBUG", "SUCCESS"])

        counts = token_counter(sample_texts, model_name, prevent_total=True)
        logger.log("List of Tokens:", counts, colors=["DEBUG", "SUCCESS"])

    logger.info("Filter text count")
    for model_name in models:
        # Whole numbers
        max_tokens = 20
        filtered_text = filter_texts(
            sample_text, model_name, max_tokens=max_tokens)
        count = token_counter(filtered_text, model_name)
        logger.log("Max Tokens:", max_tokens, colors=["GRAY", "DEBUG"])
        logger.log("New Tokens:", count, colors=["DEBUG", "SUCCESS"])
        logger.log("Filtered Text:", filtered_text,
                   colors=["DEBUG", "SUCCESS"])

    logger.info("Filter batch of texts for: list[str]")
    for model_name in models:
        # Whole numbers
        max_tokens = 20
        filtered_texts = filter_texts(
            sample_texts, model_name, max_tokens=max_tokens)
        count = token_counter(filtered_texts, model_name)
        logger.log("Max Tokens:", max_tokens, colors=["GRAY", "DEBUG"])
        logger.log("New Tokens:", count, colors=["DEBUG", "SUCCESS"])
        logger.log("Filtered Texts:", filtered_texts,
                   colors=["DEBUG", "SUCCESS"])

        # Percentage
        max_tokens = 0.5
        filtered_texts = filter_texts(
            sample_texts, model_name, max_tokens=max_tokens)
        count = token_counter(filtered_texts, model_name)
        logger.log("Max Tokens:", max_tokens, colors=["GRAY", "DEBUG"])
        logger.log("New Tokens:", count, colors=["DEBUG", "SUCCESS"])
        logger.log("Filtered Texts:", filtered_texts,
                   colors=["DEBUG", "SUCCESS"])

    logger.info("Filter batch of texts for: list[ChatMessage]")
    for model_name in models:
        # Whole numbers
        max_tokens = 20
        filtered_texts = filter_texts(
            sample_chat_messages, model_name, max_tokens=max_tokens)
        count = token_counter(filtered_texts, model_name)
        logger.log("Max Tokens:", max_tokens, colors=["GRAY", "DEBUG"])
        logger.log("New Tokens:", count, colors=["DEBUG", "SUCCESS"])
        logger.log("Filtered Texts:", filtered_texts,
                   colors=["DEBUG", "SUCCESS"])
