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

# Updated: Multi-class labeling with multiple label outputs


def multi_class_label(review, model, tokenizer, max_length=50):
    prompt = (
        f"Classify the review by selecting one or more categories: Positive, Negative, Neutral. "
        f"Return the categories as a comma-separated list (e.g., 'Positive, Neutral').\n"
        f"Examples:\n"
        f"Review: Amazing quality and fast delivery!\n"
        f"Categories: Positive\n"
        f"Review: Poor battery life and terrible service.\n"
        f"Categories: Negative\n"
        f"Review: Decent performance, nothing special.\n"
        f"Categories: Neutral\n"
        f"Review: Great features but overpriced.\n"
        f"Categories: Positive, Negative\n"
        f"Review: Works as expected, but customer service was unhelpful.\n"
        f"Categories: Neutral, Negative\n"
        f"Review: {review}\n"
        f"Categories: "
    )
    input_tokens = tokenizer.encode(
        prompt, max_length=max_length, padding=True, truncation=True)
    max_len = max_length
    input_tokens = mx.array(
        [input_tokens + [tokenizer.pad_token_id] * (max_len - len(input_tokens))])
    sampler = make_sampler(temp=0.1)
    categories = generate(model, tokenizer, prompt=prompt,
                          max_tokens=20, sampler=sampler)
    valid_categories = ["positive", "negative", "neutral"]
    categories_lower = categories.lower().strip()
    category_matches = re.findall(
        r"\b(positive|negative|neutral)\b", categories_lower)
    if not category_matches:
        return "Neutral (invalid response)"
    # Capitalize and join unique categories
    unique_categories = sorted(set(category.capitalize()
                               for category in category_matches))
    return ", ".join(unique_categories)

# Short summary (1 sentence)


def generate_short_summary(reviews, model, tokenizer, max_length=50):
    combined_reviews = " ".join(reviews)
    prompt = (
        f"Summarize the following reviews in one concise sentence:\n"
        f"Example:\n"
        f"Reviews: Great camera but slow processor.\n"
        f"Summary: The product has a great camera but a slow processor.\n"
        f"Reviews: {combined_reviews}\n"
        f"Summary: "
    )
    sampler = make_sampler(temp=0.3)
    summary = generate(model, tokenizer, prompt=prompt,
                       max_tokens=50, sampler=sampler)
    summary = re.match(r'^.*?[.?!]', summary)
    summary = summary.group(
        0) if summary else "Summary could not be generated."
    key_themes = ["delivery", "quality", "battery",
                  "customer service", "price", "features"]
    if not any(theme in summary.lower() for theme in key_themes):
        return "Summary could not be generated (missing key themes)."
    return summary.strip()

# Medium summary (2 sentences)


def generate_medium_summary(reviews, model, tokenizer, max_length=100):
    combined_reviews = " ".join(reviews)
    prompt = (
        f"Summarize the following reviews in two sentences, capturing key points:\n"
        f"Example:\n"
        f"Reviews: Great camera but slow processor. Battery life is amazing.\n"
        f"Summary: The product features an excellent camera and impressive battery life. However, its processor is notably slow, impacting performance.\n"
        f"Reviews: {combined_reviews}\n"
        f"Summary: "
    )
    sampler = make_sampler(temp=0.3)
    summary = generate(model, tokenizer, prompt=prompt,
                       max_tokens=80, sampler=sampler)
    summary_sentences = re.findall(r'[^.!?]+[.!?]', summary)
    if len(summary_sentences) < 2:
        return "Summary could not be generated (insufficient length)."
    key_themes = ["delivery", "quality", "battery",
                  "customer service", "price", "features"]
    if not any(theme in summary.lower() for theme in key_themes):
        return "Summary could not be generated (missing key themes)."
    return " ".join(summary_sentences[:2]).strip()

# Long summary (3 sentences)


def generate_long_summary(reviews, model, tokenizer, max_length=100):
    combined_reviews = " ".join(reviews)
    prompt = (
        f"Generate a detailed summary of the following reviews in three sentences, capturing all key points:\n"
        f"Example:\n"
        f"Reviews: Great camera but slow processor. Battery life is amazing.\n"
        f"Summary: The product features an excellent camera that delivers high-quality images. Its battery life is exceptional, lasting through extended use. However, the slow processor significantly hinders performance.\n"
        f"Reviews: {combined_reviews}\n"
        f"Summary: "
    )
    sampler = make_sampler(temp=0.3)
    summary = generate(model, tokenizer, prompt=prompt,
                       max_tokens=120, sampler=sampler)
    summary_sentences = re.findall(r'[^.!?]+[.!?]', summary)
    if len(summary_sentences) < 3:
        return "Summary could not be generated (insufficient length)."
    key_themes = ["delivery", "quality", "battery",
                  "customer service", "price", "features"]
    if not any(theme in summary.lower() for theme in key_themes):
        return "Summary could not be generated (missing key themes)."
    return " ".join(summary_sentences[:3]).strip()


# Real-world usage
model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
reviews = [
    "Amazing quality and fast delivery, highly recommend!",
    "Poor battery life and terrible customer service.",
    "Works as expected, nothing special but reliable.",
    "Great features but overpriced and slow response time.",
    "Good design, but the battery drains quickly and support was average."
]

# Tokenize with padding
max_length = 50
input_ids_list = [tokenizer.encode(
    review, max_length=max_length, padding=True, truncation=True) for review in reviews]
max_len = max(len(ids) for ids in input_ids_list)
input_ids = [ids + [tokenizer.pad_token_id] *
             (max_len - len(ids)) for ids in input_ids_list]
input_ids = mx.array(np.array(input_ids))
pad_token_id = tokenizer.pad_token_id
mask = create_padding_mask(input_ids, pad_token_id)

# Multi-class labeling with multiple label outputs
print("Multi-Class Labeling:")
for i, review in enumerate(reviews):
    categories = multi_class_label(review, model, tokenizer, max_length)
    print(f"Review {i+1}: {review}\nCategories: {categories}")

# Summaries
print("\nShort Summary of Reviews:")
short_summary = generate_short_summary(
    reviews, model, tokenizer, max_length=50)
print(short_summary)

print("\nMedium Summary of Reviews:")
medium_summary = generate_medium_summary(
    reviews, model, tokenizer, max_length=100)
print(medium_summary)

print("\nLong Summary of Reviews:")
long_summary = generate_long_summary(reviews, model, tokenizer, max_length=100)
print(long_summary)

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


class TestMultiClassLabeling:
    def test_multi_class_label_positive(self):
        model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
        review = "Amazing quality and fast delivery, highly recommend!"
        result = multi_class_label(review, model, tokenizer)
        expected = "Positive"
        assert result == expected, f"Expected categories {expected}, got {result}"

    def test_multi_class_label_negative(self):
        model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
        review = "Poor battery life and terrible customer service."
        result = multi_class_label(review, model, tokenizer)
        expected = "Negative"
        assert result == expected, f"Expected categories {expected}, got {result}"

    def test_multi_class_label_neutral(self):
        model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
        review = "Works as expected, nothing special but reliable."
        result = multi_class_label(review, model, tokenizer)
        expected = "Neutral"
        assert result == expected, f"Expected categories {expected}, got {result}"

    def test_multi_class_label_positive_negative(self):
        model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
        review = "Great features but overpriced and slow response time."
        result = multi_class_label(review, model, tokenizer)
        expected = "Negative, Positive"
        assert result == expected, f"Expected categories {expected}, got {result}"

    def test_multi_class_label_neutral_negative(self):
        model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
        review = "Good design, but the battery drains quickly and support was average."
        result = multi_class_label(review, model, tokenizer)
        expected = "Negative, Neutral"
        assert result == expected, f"Expected categories {expected}, got {result}"


class TestSummaries:
    def test_short_summary_length(self):
        model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
        reviews = ["Great product!", "Poor service."]
        result = generate_short_summary(reviews, model, tokenizer)
        expected_sentences = 1
        result_sentences = len(re.findall(r'[^.!?]+[.!?]', result))
        assert result_sentences == expected_sentences, f"Expected {expected_sentences} sentence, got {result_sentences}"

    def test_medium_summary_length(self):
        model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
        reviews = ["Fast delivery, great quality.", "Poor battery life."]
        result = generate_medium_summary(reviews, model, tokenizer)
        expected_sentences = 2
        result_sentences = len(re.findall(r'[^.!?]+[.!?]', result))
        assert result_sentences == expected_sentences, f"Expected {expected_sentences} sentences, got {result_sentences}"

    def test_long_summary_length(self):
        model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
        reviews = ["Fast delivery, great quality.", "Poor battery life."]
        result = generate_long_summary(reviews, model, tokenizer)
        expected_sentences = 3
        result_sentences = len(re.findall(r'[^.!?]+[.!?]', result))
        assert result_sentences == expected_sentences, f"Expected {expected_sentences} sentences, got {result_sentences}"

    def test_summary_themes(self):
        model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
        reviews = ["Fast delivery, great quality.", "Poor battery life."]
        result = generate_long_summary(reviews, model, tokenizer)
        expected_themes = ["delivery", "quality", "battery",
                           "customer service", "price", "features"]
        assert any(theme in result.lower(
        ) for theme in expected_themes), f"Expected themes {expected_themes}, got {result}"
