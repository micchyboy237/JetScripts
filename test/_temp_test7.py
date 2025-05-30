import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import numpy as np
import re

# Attention layer with padding mask for feedback analysis


class FeedbackAttention(nn.Module):
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
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(values_hat)

# Create padding mask for variable-length reviews


def create_padding_mask(input_ids, pad_token_id):
    mask = (input_ids != pad_token_id).astype(mx.float32)
    return mask[:, None, None, :]  # Shape: (batch, 1, 1, seq_len)


# Real-world usage: Sentiment labelling and summarization
model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ")
reviews = [
    "Amazing product! Fast delivery and great quality.",
    "Good features, but the battery life is disappointing.",
    "Not worth the price. Poor customer service."
]

# Tokenize with padding using tokenizer.encode
max_length = 50
input_ids_list = [tokenizer.encode(
    review, max_length=max_length, padding=True, truncation=True) for review in reviews]
max_len = max(len(ids) for ids in input_ids_list)
input_ids = [ids + [tokenizer.pad_token_id] *
             (max_len - len(ids)) for ids in input_ids_list]
input_ids = mx.array(np.array(input_ids))
pad_token_id = tokenizer.pad_token_id
mask = create_padding_mask(input_ids, pad_token_id)

# Sentiment labelling with improved prompt
print("Sentiment Analysis:")
for i, review in enumerate(reviews):
    prompt = (
        f"Classify the sentiment of the following review as positive, negative, or neutral. Return only the sentiment label.\n"
        f"Example:\n"
        f"Review: Great product, highly recommend!\n"
        f"Sentiment: Positive\n"
        f"Review: Terrible experience, never again.\n"
        f"Sentiment: Negative\n"
        f"Review: {review}\n"
        f"Sentiment: "
    )
    input_tokens = tokenizer.encode(
        prompt, max_length=max_length, padding=True, truncation=True)
    input_tokens = mx.array(
        [input_tokens + [tokenizer.pad_token_id] * (max_len - len(input_tokens))])
    # Very low temperature for deterministic output
    sampler = make_sampler(temp=0.1)
    sentiment = generate(model, tokenizer, prompt=prompt,
                         max_tokens=10, sampler=sampler)
    # Validate sentiment
    valid_sentiments = ["positive", "negative", "neutral"]
    sentiment_lower = sentiment.lower().strip()
    sentiment_match = re.search(
        r"\b(positive|negative|neutral)\b", sentiment_lower)
    sentiment = sentiment_match.group(1).capitalize(
    ) if sentiment_match else "Neutral (invalid response)"
    print(f"Review {i+1}: {review}\nSentiment: {sentiment}")

# Summarization with improved prompt
combined_reviews = " ".join(reviews)
summary_prompt = (
    f"Summarize the following reviews in one sentence, capturing all key points:\n"
    f"Example:\n"
    f"Reviews: Great camera but slow processor. Battery life is amazing.\n"
    f"Summary: The product has a great camera and battery life but a slow processor.\n"
    f"Reviews: {combined_reviews}\n"
    f"Summary: "
)
summary_tokens = tokenizer.encode(
    summary_prompt, max_length=max_length, padding=True, truncation=True)
sampler = make_sampler(temp=0.3)  # Lower temperature for coherence
summary = generate(model, tokenizer, prompt=summary_prompt,
                   max_tokens=50, sampler=sampler)
# Extract first sentence and validate content
summary = re.match(r'^.*?[.?!]', summary)
summary = summary.group(0) if summary else "Summary could not be generated."
# Check for key themes
key_themes = ["delivery", "quality", "battery", "customer service"]
if not any(theme in summary.lower() for theme in key_themes):
    summary = "Summary could not be generated (missing key themes)."
print("\nSummary of Reviews:")
print(summary)

# Pytest tests


class TestFeedbackPaddingMask:
    def test_padding_mask_shape(self):
        input_ids = mx.array([[1, 2, 0], [1, 2, 3], [1, 0, 0]])
        pad_token_id = 0
        result = create_padding_mask(input_ids, pad_token_id)
        expected_shape = (3, 1, 1, 3)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    def test_padding_mask_values(self):
        input_ids = mx.array([[1, 2, 0], [1, 2, 3]])
        pad_token_id = 0
        result = create_padding_mask(input_ids, pad_token_id)
        expected = mx.array([[[[1., 1., 0.]]], [[[1., 1., 1.]]]])
        assert mx.allclose(result, expected), "Padding mask values incorrect"
