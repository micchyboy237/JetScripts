from jet.llm.mlx.helpers import load_model
from jet.llm.mlx.mlx_types import LLMModelType
import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate

# Simplified attention layer with causal mask


class SimpleAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

    def __call__(self, queries, keys, values, mask=None):
        B, L, D = queries.shape
        queries = self.query_proj(queries).reshape(
            B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = self.key_proj(keys).reshape(
            B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        values = self.value_proj(values).reshape(
            B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        scale = mx.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(values_hat)

# Create causal mask


def create_causal_mask(seq_len: int):
    mask = mx.triu(mx.ones((seq_len, seq_len)) * float('-inf'), k=1)
    return mask

# Labeling function for generated text


def label_generated_text(text: str) -> str:
    """Assigns a sentiment label to generated text based on keywords."""
    positive_keywords = ["happy", "great", "wonderful", "beautiful"]
    negative_keywords = ["sad", "terrible", "awful", "bad"]
    text = text.lower()
    if any(keyword in text for keyword in positive_keywords):
        return "positive"
    elif any(keyword in text for keyword in negative_keywords):
        return "negative"
    return "neutral"

# Summarization function


def summarize_text(model, tokenizer, text: str, max_summary_tokens: int) -> str:
    """Generates a summary of the input text with a specified token limit."""
    prompt = f"Summarize the following text in a concise manner:\n{text}\nSummary:"
    summary = generate(model, tokenizer, prompt=prompt,
                       max_tokens=max_summary_tokens, verbose=False)
    return summary.strip()


# Example usage
# Load model and tokenizer
llm_model: LLMModelType = "qwen3-1.7b-4bit"
model, tokenizer = load_model(llm_model)
prompt = "Once upon a time"

# Token budget
MAX_TOTAL_TOKENS = 20
MAX_LABEL_TOKENS = 2  # Max tokens for label (e.g., "[positive]")
MAX_SUMMARY_TOKENS = 5  # Tokens for summary
MAX_TEXT_TOKENS = MAX_TOTAL_TOKENS - MAX_LABEL_TOKENS - \
    MAX_SUMMARY_TOKENS  # Reserve tokens

# Use encode instead of calling the tokenizer
input_ids = tokenizer.encode(prompt)
input_ids = mx.array([input_ids])  # Add batch dimension

seq_len = input_ids.shape[1]
mask = create_causal_mask(seq_len)

# Generate text
response = generate(model, tokenizer, prompt=prompt,
                    max_tokens=MAX_TEXT_TOKENS, verbose=False)
label = label_generated_text(response)
summary = summarize_text(model, tokenizer, response,
                         max_summary_tokens=MAX_SUMMARY_TOKENS)
final_output = f"{response} [{label}]"
print(f"Generated text: {response}")
print(f"Assigned label: {label}")
print(f"Summary: {summary}")
print(f"Total tokens (text + label): {len(tokenizer.encode(final_output))}")
print(f"Summary tokens: {len(tokenizer.encode(summary))}")

# Pytest tests


class TestCausalMask:
    def test_causal_mask_shape(self):
        seq_len = 5
        expected = mx.triu(mx.ones((seq_len, seq_len)) * float('-inf'), k=1)
        result = create_causal_mask(seq_len)
        assert result.shape == expected.shape, f"Expected shape {expected.shape}, got {result.shape}"
        assert mx.all(result == expected), "Causal mask values incorrect"

    def test_causal_mask_values(self):
        seq_len = 3
        expected = mx.array([[0., -float('inf'), -float('inf')],
                             [0., 0., -float('inf')],
                             [0., 0., 0.]])
        result = create_causal_mask(seq_len)
        assert mx.allclose(
            result, expected), "Causal mask values not close to expected"


class TestLabelGeneratedText:
    def test_label_positive(self):
        input_text = "It was a wonderful day in the beautiful town."
        expected = "positive"
        result = label_generated_text(input_text)
        assert result == expected, f"Expected label {expected}, got {result}"

    def test_label_negative(self):
        input_text = "The town was struck by a terrible storm."
        expected = "negative"
        result = label_generated_text(input_text)
        assert result == expected, f"Expected label {expected}, got {result}"

    def test_label_neutral(self):
        input_text = "The town was quiet and peaceful."
        expected = "neutral"
        result = label_generated_text(input_text)
        assert result == expected, f"Expected label {expected}, got {result}"


class TestSummarization:
    def test_summary_length(self, monkeypatch):
        # Mock generate to return a fixed summary
        def mock_generate(*args, **kwargs):
            return "Quiet town in 19th century."
        monkeypatch.setattr("mlx_lm.generate", mock_generate)

        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return [1] * len(text.split())  # Approximate token count
        mock_tokenizer = MockTokenizer()

        # Test summarization
        input_text = "The town was quiet and peaceful in the 19th century."
        max_summary_tokens = 5
        expected = max_summary_tokens
        result = len(mock_tokenizer.encode(summarize_text(
            None, mock_tokenizer, input_text, max_summary_tokens)))
        assert result <= expected, f"Summary tokens {result} exceed limit {expected}"


class TestTokenBudget:
    def test_token_budget(self, monkeypatch):
        # Mock generate to return fixed responses
        def mock_generate(*args, **kwargs):
            if kwargs["prompt"].startswith("Summarize"):
                return "Quiet town"
            return "The town was quiet"
        monkeypatch.setattr("mlx_lm.generate", mock_generate)

        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                if text == "Once upon a time":
                    return [1, 2, 3, 4]  # 4 tokens
                return [1] * len(text.split())  # Approximate token count
        mock_tokenizer = MockTokenizer()

        # Run generation, labeling, and summarization
        MAX_TOTAL_TOKENS = 20
        MAX_LABEL_TOKENS = 2
        MAX_SUMMARY_TOKENS = 5
        MAX_TEXT_TOKENS = MAX_TOTAL_TOKENS - MAX_LABEL_TOKENS - MAX_SUMMARY_TOKENS
        response = mock_generate(
            None, None, prompt="Once upon a time", max_tokens=MAX_TEXT_TOKENS)
        label = label_generated_text(response)
        final_output = f"{response} [{label}]"
        total_tokens = len(mock_tokenizer.encode(final_output))
        expected = MAX_TOTAL_TOKENS - MAX_SUMMARY_TOKENS
        result = total_tokens
        assert result <= expected, f"Total tokens {result} exceeds budget {expected}"
