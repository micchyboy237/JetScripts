import json
from jet.token.token_utils import token_counter, filter_texts
from jet.logger import logger
from jet.transformers.object import make_serializable
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
        role=MessageRole.SYSTEM,
        content="You are a helpful AI assistant focused on answering questions about programming and software development."
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="Can you help me understand how Python list comprehensions work? I find them confusing."
    ),
    ChatMessage(
        role=MessageRole.ASSISTANT,
        content="List comprehensions are a concise way to create lists in Python. They follow this basic syntax: [expression for item in iterable]. For example, to create a list of squares from 1-5, you can write: squares = [x**2 for x in range(1,6)]"
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="That makes sense! Could you show another example with filtering using an if condition?"
    ),
    ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Here's an example of filtering with list comprehension: [x for x in range(10) if x % 2 == 0] will create a list of even numbers from 0-9. The if condition only includes items that match the criteria."
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="What about using list comprehensions with strings? Can you give an example?"
    ),
]

if __name__ == "__main__":
    models = ["qwen3"]
    default_max_tokens = 20
    max_tokens = default_max_tokens

    logger.info("\n\nCount tokens for newlines")
    for model_name in models:
        logger.newline()
        logger.log("Model:", model_name, colors=["WHITE", "PURPLE"])
        count1 = token_counter("\n", model_name)
        count2 = token_counter("\n\n", model_name)
        count3 = token_counter("\n\n\n", model_name)
        logger.log("Tokens:", [count1, count2, count3],
                   colors=["GRAY", "SUCCESS"])

    logger.info("\n\nCount tokens for: str")
    for model_name in models:
        logger.newline()
        logger.log("Model:", model_name, colors=["WHITE", "PURPLE"])
        count = token_counter(sample_text, model_name)
        logger.log("Tokens:", count, colors=["GRAY", "SUCCESS"])

    logger.info("\n\nCount batch tokens for: list[str]")
    for model_name in models:
        logger.newline()
        logger.log("Model:", model_name, colors=["WHITE", "PURPLE"])
        count = token_counter(sample_texts, model_name)
        logger.log("Tokens:", count, colors=["GRAY", "SUCCESS"])

        counts = token_counter(sample_texts, model_name, prevent_total=True)
        logger.log("List of Tokens:", counts, colors=["GRAY", "SUCCESS"])

    logger.info(f"\n\nFilter text count (max: {max_tokens})")
    for model_name in models:
        logger.newline()
        logger.log("Model:", model_name, colors=["WHITE", "PURPLE"])
        # Whole numbers
        filtered_text = filter_texts(
            sample_text, model_name, max_tokens=max_tokens)
        count = token_counter(filtered_text, model_name)
        orig_count = token_counter(sample_text, model_name)

        logger.log("Orig Tokens:", orig_count, colors=["GRAY", "DEBUG"])
        logger.log("New Tokens:", count, colors=["GRAY", "SUCCESS"])
        logger.log("Filtered Text:", filtered_text,
                   colors=["GRAY", "SUCCESS"])

    logger.info(
        f"\n\nFilter batch of texts for: list[str] (max: {max_tokens})")
    for model_name in models:
        logger.newline()
        logger.log("Model:", model_name, colors=["WHITE", "PURPLE"])

        # Whole numbers
        max_tokens = default_max_tokens
        filtered_texts = filter_texts(
            sample_texts, model_name, max_tokens=max_tokens)
        count = token_counter(str(filtered_texts), model_name)
        orig_count = token_counter(str(sample_texts), model_name)

        logger.log("Max Tokens:", max_tokens, colors=["GRAY", "DEBUG"])
        logger.log("Orig Tokens:", orig_count, colors=["GRAY", "DEBUG"])
        logger.log("New Tokens:", count, colors=["GRAY", "SUCCESS"])
        logger.log("Filtered Texts:", len(filtered_texts),
                   colors=["GRAY", "SUCCESS"])

        # Percentage
        max_tokens = 0.5
        filtered_texts = filter_texts(
            sample_texts, model_name, max_tokens=max_tokens)
        count = token_counter(str(filtered_texts), model_name)
        orig_count = token_counter(str(sample_texts), model_name)

        logger.newline()
        logger.log("Max Tokens:", max_tokens, colors=["GRAY", "DEBUG"])
        logger.log("Orig Tokens:", orig_count, colors=["GRAY", "DEBUG"])
        logger.log("New Tokens:", count, colors=["GRAY", "SUCCESS"])
        logger.log("Filtered Texts:", len(filtered_texts),
                   colors=["GRAY", "SUCCESS"])

    logger.info(
        f"\n\nFilter batch of texts for: list[ChatMessage] (max: {max_tokens})")
    for model_name in models:
        logger.newline()
        logger.log("Model:", model_name, colors=["WHITE", "PURPLE"])
        # Whole numbers
        filtered_texts = filter_texts(
            sample_chat_messages, model_name, max_tokens=max_tokens)
        count = token_counter(str(filtered_texts), model_name)
        orig_count = token_counter(str(sample_chat_messages), model_name)

        logger.log("Orig Tokens:", orig_count, colors=["GRAY", "DEBUG"])
        logger.log("New Tokens:", count, colors=["GRAY", "SUCCESS"])
        logger.log("Filtered Texts:", len(filtered_texts),
                   colors=["GRAY", "SUCCESS"])

    logger.info(
        f"\n\nFilter batch of texts for: list[dict] (max: {max_tokens})")
    sample_chat_messages_list_dict = make_serializable(sample_chat_messages)
    for model_name in models:
        logger.newline()
        logger.log("Model:", model_name, colors=["WHITE", "PURPLE"])
        # Whole numbers
        filtered_texts = filter_texts(
            sample_chat_messages, model_name, max_tokens=max_tokens)
        count = token_counter(str(filtered_texts), model_name)
        orig_count = token_counter(str(sample_chat_messages), model_name)

        logger.log("Orig Tokens:", orig_count, colors=["GRAY", "DEBUG"])
        logger.log("New Tokens:", count, colors=["GRAY", "SUCCESS"])
        logger.log("Filtered Texts:", len(filtered_texts),
                   colors=["GRAY", "SUCCESS"])
