# run_count_tokens.py

from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_CONTEXTS
from jet.adapters.llama_cpp.tokens import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_TYPES

MODEL: LLAMACPP_LLM_TYPES = "qwen3-instruct-2507:4b"
MODEL_MAX_CONTEXT: int = LLAMACPP_MODEL_CONTEXTS[MODEL]


def demo_plain_text_single() -> None:
    """
    Demo: counting tokens for a single plain string.
    - No chat framing
    - No role overhead
    """
    text = "Hello world, this is a test."
    tokens = count_tokens(text, model=MODEL)
    print("demo_plain_text_single:", tokens)


def demo_plain_text_batch() -> None:
    """
    Demo: counting tokens for a list of strings.
    - Returns total token count by default
    """
    texts = ["Hello world", "How are you today?", "This is a longer sentence."]
    tokens = count_tokens(texts, model=MODEL)
    print("demo_plain_text_batch:", tokens)


def demo_plain_text_batch_per_item() -> None:
    """
    Demo: counting tokens per item in a batch.
    - Uses prevent_total=True
    """
    texts = ["Hello world", "How are you today?", "This is a longer sentence."]
    tokens = count_tokens(
        texts,
        model=MODEL,
        prevent_total=True,
    )
    print("demo_plain_text_batch_per_item:", tokens)


def demo_dict_content_only() -> None:
    """
    Demo: legacy dict input.
    - Only 'content' is counted
    - Role is ignored
    """
    message = {"content": "Translate this sentence to Japanese."}
    tokens = count_tokens(message, model=MODEL)
    print("demo_dict_content_only:", tokens)


def demo_chat_messages() -> None:
    """
    Demo: chat-style messages.
    - Uses model chat template
    - Includes roles, separators, BOS/EOS
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain tokenization in simple terms."},
    ]
    tokens = count_tokens(messages, model=MODEL)
    print("demo_chat_messages:", tokens)


def demo_chat_budgeting() -> None:
    """
    Demo: prompt budgeting for chat models.
    - Counts everything before generation
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion."},
    ]

    prompt_tokens = count_tokens(messages, model=MODEL)
    max_context = 8192
    max_context = max_context if MODEL_MAX_CONTEXT > max_context else MODEL_MAX_CONTEXT
    max_new_tokens = max_context - prompt_tokens

    print("demo_chat_budgeting:")
    print("  prompt_tokens:", prompt_tokens)
    print("  max_new_tokens:", max_new_tokens)


def demo_chat_with_tiktoken_error() -> None:
    """
    Demo: showing safe failure when chat messages are used
    without a model-specific tokenizer.
    """
    messages = [{"role": "user", "content": "Hello!"}]

    try:
        count_tokens(messages)
    except ValueError as e:
        print("demo_chat_with_tiktoken_error:", str(e))


def demo_chat_with_tools() -> None:
    """
    Demo: chat messages with tool calls.
    - Tool payloads are serialized and counted
    """
    messages = [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_weather",
                    "arguments": {"city": "Paris"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "It is 18°C and sunny.",
        },
    ]

    tokens = count_tokens(messages, model=MODEL)
    print("demo_chat_with_tools:", tokens)


def main() -> None:
    demo_plain_text_single()
    demo_plain_text_batch()
    demo_plain_text_batch_per_item()
    demo_dict_content_only()
    demo_chat_messages()
    demo_chat_budgeting()
    demo_chat_with_tiktoken_error()
    demo_chat_with_tools()


if __name__ == "__main__":
    main()
