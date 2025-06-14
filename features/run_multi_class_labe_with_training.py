import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import logging
from sentence_transformers import SentenceTransformer
from typing import List, TypedDict, Literal, Optional
import re
import numpy as np
from mlx_lm.sample_utils import make_sampler
from mlx_lm import load, generate
import mlx.nn as nn
import mlx.core as mx
import pytest
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def create_padding_mask(input_ids, pad_token_id):
    mask = (input_ids != pad_token_id).astype(mx.float32)
    return mask[:, None, None, :]


class ClassificationResult(TypedDict):
    category: Literal["Positive", "Negative", "Neutral", "Mixed"]
    confidence: float


def embed_reviews(reviews: List[str], embedder: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    """Embed reviews in batches using SentenceTransformer, optimized for Mac M1."""
    logger.info("Embedding %d reviews in batches with batch_size=%d",
                len(reviews), batch_size)
    device = "cpu"
    logger.info("Using device: %s", device)
    try:
        embeddings = embedder.encode(
            reviews,
            batch_size=batch_size,
            convert_to_numpy=True,
            device=device,
            show_progress_bar=True
        )
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        logger.debug("Embeddings shape: %s, dtype: %s",
                     embeddings.shape, embeddings.dtype)
        return embeddings
    except Exception as e:
        logger.error("Error embedding reviews: %s", str(e))
        raise RuntimeError(f"Failed to embed reviews: {str(e)}")


def train_classifier() -> tuple[LogisticRegression, LabelEncoder]:
    """Train a simple classifier on example reviews for multi-class labeling."""
    logger.info("Training logistic regression classifier")
    example_reviews = [
        "Fantastic product, highly recommend!",
        "Terrible quality, broke after one use.",
        "Standard performance, nothing special.",
        "Great features but overpriced."
    ]
    labels = ["Positive", "Negative", "Neutral", "Mixed"]

    embedder = SentenceTransformer(
        "static-retrieval-mrl-en-v1", backend="onnx")
    embeddings = embed_reviews(example_reviews, embedder)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    classifier = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=200)
    classifier.fit(embeddings, encoded_labels)

    logger.info("Classifier trained successfully")
    return classifier, label_encoder


def multi_class_label(
    review: str,
    classifier: LogisticRegression,
    label_encoder: LabelEncoder,
    embedder: SentenceTransformer,
    max_length: int = 50
) -> ClassificationResult:
    """Classify a review using embeddings and a trained classifier."""
    logger.info("Classifying review: %s", review[:50])
    try:
        # Truncate review if necessary
        if len(review) > max_length:
            review = review[:max_length]
            logger.debug("Truncated review to %d characters", max_length)

        # Generate embedding
        embedding = embed_reviews([review], embedder)[0]
        logger.debug("Embedding generated, shape: %s", embedding.shape)

        # Predict category
        pred_proba = classifier.predict_proba([embedding])[0]
        pred_index = np.argmax(pred_proba)
        confidence = float(pred_proba[pred_index])
        category = label_encoder.inverse_transform([pred_index])[0]

        logger.info("Predicted category: %s, confidence: %.4f",
                    category, confidence)
        return {"category": category, "confidence": confidence}
    except Exception as e:
        logger.error("Error in classification: %s", str(e))
        return {"category": "Neutral", "confidence": 0.0}


def generate_short_summary(reviews: List[str], model, tokenizer, max_length: int = 50) -> str:
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
    summary_match = re.match(r'^.*?[.?!]', summary)
    summary = summary_match.group(
        0) if summary_match else "Summary could not be generated."
    key_themes = ["delivery", "quality", "battery",
                  "customer service", "price", "features"]
    if not any(theme in summary.lower() for theme in key_themes):
        return "Summary could not be generated (missing key themes)."
    return summary.strip()


def generate_medium_summary(reviews: List[str], model, tokenizer, max_length: int = 100) -> str:
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


def generate_long_summary(reviews: List[str], model, tokenizer, max_length: int = 100) -> str:
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


# Initialize model and resources
model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
embedder = SentenceTransformer("static-retrieval-mrl-en-v1", backend="onnx")
classifier, label_encoder = train_classifier()

# Example usage
reviews = [
    "Fantastic product! Fast delivery and excellent quality.",
    "Good features, but the battery life is disappointing and overpriced.",
    "Not worth the price. Poor customer service and average performance.",
    "Standard product, nothing remarkable but works as expected."
]
max_length = 50

print("Multi-Class Labeling:")
for i, review in enumerate(reviews):
    result = multi_class_label(
        review, classifier, label_encoder, embedder, max_length)
    print(
        f"Review {i+1}: {review}\nCategory: {result['category']} (Confidence: {result['confidence']:.4f})")

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

# Updated tests


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
    @pytest.fixture(autouse=True)
    def setup_classifier(self):
        self.embedder = SentenceTransformer(
            "static-retrieval-mrl-en-v1", backend="onnx")
        self.classifier, self.label_encoder = train_classifier()

    def test_multi_class_label_positive(self):
        review = "Fantastic product, highly recommend!"
        result = multi_class_label(
            review, self.classifier, self.label_encoder, self.embedder)
        expected = {"category": "Positive", "confidence": float}
        assert result["category"] == expected[
            "category"], f"Expected category {expected['category']}, got {result['category']}"
        assert isinstance(result["confidence"],
                          float), "Confidence should be a float"

    def test_multi_class_label_negative(self):
        review = "Terrible quality, broke after one use."
        result = multi_class_label(
            review, self.classifier, self.label_encoder, self.embedder)
        expected = {"category": "Negative", "confidence": float}
        assert result["category"] == expected[
            "category"], f"Expected category {expected['category']}, got {result['category']}"
        assert isinstance(result["confidence"],
                          float), "Confidence should be a float"

    def test_multi_class_label_mixed(self):
        review = "Great features but overpriced."
        result = multi_class_label(
            review, self.classifier, self.label_encoder, self.embedder)
        expected = {"category": "Mixed", "confidence": float}
        assert result["category"] == expected[
            "category"], f"Expected category {expected['category']}, got {result['category']}"
        assert isinstance(result["confidence"],
                          float), "Confidence should be a float"

    def test_multi_class_label_neutral(self):
        review = "Standard performance, nothing special."
        result = multi_class_label(
            review, self.classifier, self.label_encoder, self.embedder)
        expected = {"category": "Neutral", "confidence": float}
        assert result["category"] == expected[
            "category"], f"Expected category {expected['category']}, got {result['category']}"
        assert isinstance(result["confidence"],
                          float), "Confidence should be a float"


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
