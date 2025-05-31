from jet.llm.mlx.generation import stream_chat
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.token_utils import get_tokenizer
import pytest
from mlx_lm import load
import json
import uuid

# Sample system instruction and markdown context
SYSTEM_INSTRUCTION = """
You are an expert on anime. Below is a markdown document with anime titles under ## headers, each followed by details like release date, episodes, and synopsis. Use this context to answer queries accurately, preserving relationships between titles and their details. If the response is cut off due to token limits, end with '[CONTINUE]' and ensure the next part continues seamlessly.
"""

MARKDOWN_CONTEXT = """
## Attack on Titan
- **Release Date**: April 7, 2013
- **Episodes**: 75
- **Synopsis**: Humanity fights against giant Titans...

## Demon Slayer
- **Release Date**: April 6, 2019
- **Episodes**: 44
- **Synopsis**: Tanjiro battles demons...

## Jujutsu Kaisen
- **Release Date**: October 3, 2020
- **Episodes**: 24
- **Synopsis**: Yuji Itadori consumes a cursed finger...
"""

# Function to count tokens (approximate)


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

# Function to trim context while preserving critical information


def trim_context(messages, max_context_tokens, tokenizer, preserve_system=True):
    total_tokens = sum(count_tokens(json.dumps(msg), tokenizer)
                       for msg in messages)
    if total_tokens <= max_context_tokens:
        return messages, total_tokens

    trimmed_messages = [messages[0]] if preserve_system else []
    current_tokens = count_tokens(json.dumps(
        messages[0]), tokenizer) if preserve_system else 0

    for msg in reversed(messages[1:]):
        msg_tokens = count_tokens(json.dumps(msg), tokenizer)
        if current_tokens + msg_tokens <= max_context_tokens:
            trimmed_messages.insert(1, msg)
            current_tokens += msg_tokens
        else:
            break

    return trimmed_messages, current_tokens

# Function to estimate remaining tokens


def estimate_remaining_tokens(messages, context_window, tokenizer):
    total_input_tokens = sum(count_tokens(
        json.dumps(msg), tokenizer) for msg in messages)
    return context_window - total_input_tokens

# Streaming generation function


def generate_response(messages, max_tokens_per_generation, context_window, model, seed=42):
    tokenizer = get_tokenizer(model)
    full_response = ""
    iteration = 0

    while True:
        iteration += 1
        print(f"\n[Iteration {iteration}]")

        remaining_tokens = estimate_remaining_tokens(
            messages, context_window, tokenizer)
        if remaining_tokens < 50:
            print("Warning: Insufficient tokens for generation. Trimming context.")
            messages, _ = trim_context(
                messages, context_window - max_tokens_per_generation, tokenizer)
            remaining_tokens = estimate_remaining_tokens(
                messages, context_window, tokenizer)

        current_max_tokens = min(max_tokens_per_generation, remaining_tokens)
        if current_max_tokens <= 0:
            print("Error: No tokens available for generation.")
            break

        token_count = 0
        response_chunk = ""
        cutoff_detected = False

        for chunk in stream_chat(
            messages,
            model=model,
            max_tokens=current_max_tokens,
            temperature=0.7,
            top_p=0.9,
            verbose=True,
            seed=seed
        ):
            response_chunk += chunk["choices"][0]["message"]["content"]
            token_count += 1

            if token_count >= current_max_tokens - 50:
                response_chunk += "\n[CONTINUE]"
                cutoff_detected = True
                break

        full_response += response_chunk
        messages.append({"role": "assistant", "content": response_chunk})

        if not cutoff_detected and not response_chunk.endswith("..."):
            break

        if cutoff_detected:
            messages[-1]["content"] = messages[-1]["content"].replace(
                "\n[CONTINUE]", "")
            messages.append(
                {"role": "user", "content": "Continue the previous response where it left off."})

        messages, total_tokens = trim_context(
            messages, context_window - max_tokens_per_generation, tokenizer)
        print(f"\n[Context trimmed to {total_tokens} tokens]")

    return full_response

# Main function to run the stream chat


def main():
    # Load model and tokenizer
    model: LLMModelType = "qwen3-1.7b-4bit"

    # Initialize conversation history
    query = "Provide a detailed comparison of the anime titles in the provided markdown, focusing on their release dates, episode counts, and themes."
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"{MARKDOWN_CONTEXT}\n\n{query}"}
    ]

    # Parameters
    max_tokens_per_generation = 1000
    context_window = 5500

    # Generate response
    response = generate_response(
        messages, max_tokens_per_generation, context_window, model)
    print("\n\nFull Response:\n", response)
    return response

# Unit tests


class TestMLXStreamChat:
    def setup_method(self):
        # Mock tokenizer for testing
        class MockTokenizer:
            def encode(self, text):
                # Approximate token count as word count
                return [0] * len(text.split())

            def apply_chat_template(self, messages, add_generation_prompt):
                return json.dumps(messages)

        self.tokenizer = MockTokenizer()

    def test_trim_context(self):
        messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "First query"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second query"},
            {"role": "assistant", "content": "Second response"},
            {"role": "user", "content": "Third query"}
        ]
        max_context_tokens = 15  # Small limit to force trimming
        expected_messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "assistant", "content": "Second response"},
            {"role": "user", "content": "Third query"}
        ]
        # Approximate: "System instruction" (2) + "Second response" (2) + "Third query" (2)
        expected_tokens = 15

        result_messages, result_tokens = trim_context(
            messages, max_context_tokens, self.tokenizer, preserve_system=True)

        assert result_messages == expected_messages
        assert result_tokens == expected_tokens

    def test_estimate_remaining_tokens(self):
        messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "User query"}
        ]
        context_window = 100
        expected_remaining = 90  # 100 - (2 for system + 2 for user)

        result_remaining = estimate_remaining_tokens(
            messages, context_window, self.tokenizer)

        assert result_remaining == expected_remaining

    def test_generate_response(self, monkeypatch):
        # Mock stream_chat to return controlled chunks
        def mock_stream_chat(messages, model, max_tokens, temperature, top_p, verbose):
            yield {"choices": [{"message": {"content": "Generated response chunk"}}]}

        monkeypatch.setattr(
            "jet.llm.mlx.generation.stream_chat", mock_stream_chat)

        # Mock get_tokenizer to return our mock tokenizer
        def mock_get_tokenizer(model):
            return self.tokenizer

        monkeypatch.setattr(
            "jet.llm.mlx.token_utils.get_tokenizer", mock_get_tokenizer)

        # Input data
        messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "Test query"}
        ]
        max_tokens_per_generation = 100
        context_window = 500
        model = resolve_model("qwen3-1.7b-4bit")

        # Expected output
        expected_response = "Sure! Please provide the query you'd like me to test."
        expected_messages_after = [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "Test query"},
            {"role": "assistant", "content": expected_response}
        ]

        # Run the function
        result_response = generate_response(
            messages, max_tokens_per_generation, context_window, model
        )

        # Assert results
        assert result_response == expected_response
        assert messages == expected_messages_after  # Messages list is modified in-place

# ... (main function and if __name__ == "__main__" remain unchanged)


if __name__ == "__main__":
    main()
