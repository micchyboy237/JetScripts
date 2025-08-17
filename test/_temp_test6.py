from jet.llm.mlx.helpers import load_model
from jet.models.model_types import LLMModelType
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
            scores = scores + mask  # Apply causal mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(values_hat)

# Create causal mask


def create_causal_mask(seq_len: int):
    mask = mx.triu(mx.ones((seq_len, seq_len)) * float('-inf'), k=1)
    return mask


# Example usage
# Load model and tokenizer
llm_model: LLMModelType = "qwen3-1.7b-4bit"
model, tokenizer = load_model(llm_model)
prompt = "Once upon a time"

# Use encode instead of calling the tokenizer
input_ids = tokenizer.encode(prompt)
input_ids = mx.array([input_ids])  # Add batch dimension

seq_len = input_ids.shape[1]
mask = create_causal_mask(seq_len)

# Generate text
response = generate(model, tokenizer, prompt=prompt,
                    max_tokens=20, verbose=True)
print(response)


# Pytest test


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
