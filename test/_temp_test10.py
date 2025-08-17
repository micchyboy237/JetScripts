from jet.llm.mlx.helpers import load_model
from jet.models.model_types import LLMModelType
import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import numpy as np

# Simplified attention with padding mask


class PaddedAttention(nn.Module):
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
            scores = mx.where(mask, scores, float('-inf'))
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(values_hat)

# Model with labeling, summarization, NER, and QA


class TextProcessor(nn.Module):
    def __init__(self, dims: int, num_heads: int, vocab_size: int, num_entities: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dims)
        self.attention = PaddedAttention(dims, num_heads)
        self.norm = nn.LayerNorm(dims)
        # Sentiment labeling (sequence-level)
        self.label_head = nn.Linear(dims, 1)
        # NER (token-level)
        self.ner_head = nn.Linear(dims, num_entities)  # e.g., O, PER, ORG, LOC
        # QA (start and end positions)
        self.qa_start_head = nn.Linear(dims, 1)
        self.qa_end_head = nn.Linear(dims, 1)

    def __call__(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        x = self.attention(x, x, x, mask)
        x = self.norm(x)

        if mask is not None:
            # Shape: (batch_size, seq_len)
            flat_mask = mask.squeeze(axis=(1, 2))
            seq_lengths = flat_mask.sum(axis=-1)  # Shape: (batch_size,)
            # (batch_size, dims)
            masked_sum = (x * flat_mask[..., None]).sum(axis=1)
        else:
            seq_lengths = mx.full((input_ids.shape[0],), x.shape[1])
            masked_sum = x.sum(axis=1)

        seq_lengths = mx.where(seq_lengths == 0, 1, seq_lengths)
        label_logits = self.label_head(
            masked_sum / seq_lengths[:, None])  # (batch_size, 1)

        ner_logits = self.ner_head(x)
        qa_start_logits = self.qa_start_head(x).squeeze(-1)
        qa_end_logits = self.qa_end_head(x).squeeze(-1)
        return x, label_logits, ner_logits, qa_start_logits, qa_end_logits


# Create padding mask


def create_padding_mask(input_ids, pad_token_id):
    mask = (input_ids != pad_token_id).astype(mx.float32)
    return mask[:, None, None, :]  # Shape: (batch_size, 1, 1, seq_len)


# Example usage
llm_model: LLMModelType = "qwen3-1.7b-4bit"
model, tokenizer = load_model(llm_model)
processor = TextProcessor(dims=512, num_heads=8,
                          vocab_size=tokenizer.vocab_size, num_entities=4)

sentences = ["Elon Musk visited Paris.", "Apple released a new product."]
questions = ["Who visited Paris?", "What did Apple release?"]
input_ids_np = tokenizer.encode(sentences, return_tensors="np", padding=True)
input_ids = mx.array(input_ids_np.tolist())
pad_token_id = tokenizer.pad_token_id
mask = create_padding_mask(input_ids, pad_token_id)

# Debug shapes
print(f"Input IDs shape: {input_ids.shape}")
print(f"Mask shape: {mask.shape}")

# Process input
encoded, label_logits, ner_logits, qa_start_logits, qa_end_logits = processor(
    input_ids, mask)
print(f"Label logits shape: {label_logits.shape}")
sentiments = mx.sigmoid(label_logits)  # Sentiment scores
print(f"Sentiments shape: {sentiments.shape}")
ner_preds = mx.argmax(ner_logits, axis=-1)  # NER predictions
qa_starts = mx.argmax(qa_start_logits, axis=-1)
qa_ends = mx.argmax(qa_end_logits, axis=-1)

# Generate summaries and QA answers
for i, (seq, question) in enumerate(zip(input_ids, questions)):
    prompt = tokenizer.decode(seq.tolist())
    # Summarization
    summary = generate(model, tokenizer, prompt=prompt, max_tokens=10)
    # Sentiment
    sentiment_score = mx.squeeze(sentiments[i]).item()  # Ensure scalar
    print(
        f"Sentiment[{i}] shape: {sentiments[i].shape}, value: {sentiments[i]}")
    sentiment = "Positive" if sentiment_score > 0.5 else "Negative"
    # NER (example mapping: 0=O, 1=PER, 2=ORG, 3=LOC)
    ner_labels = ["O", "PER", "ORG", "LOC"]
    ner_tokens = [ner_labels[p.item()] for p in ner_preds[i]]
    # QA
    start, end = qa_starts[i].item(), qa_ends[i].item() + 1
    answer_tokens = seq[start:end].tolist()
    answer = tokenizer.decode(answer_tokens)
    print(f"Sentence: {prompt}\nSentiment: {sentiment} ({sentiment_score:.3f})\nSummary: {summary}\nNER: {ner_tokens}\nQ: {question}\nA: {answer}\n")

# Pytest tests


class TestTextProcessor:
    def test_padding_mask_shape(self):
        input_ids = mx.array([[1, 2, 0], [1, 2, 3]])
        pad_token_id = 0
        result = create_padding_mask(input_ids, pad_token_id)
        expected_shape = (2, 1, 1, 3)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    def test_labeling_output(self):
        processor = TextProcessor(dims=64, num_heads=4, vocab_size=100)
        input_ids = mx.array([[1, 2, 0], [1, 2, 3]])
        pad_token_id = 0
        mask = create_padding_mask(input_ids, pad_token_id)
        _, result, _, _, _ = processor(input_ids, mask)
        expected_shape = (2, 1)  # Batch size 2, binary label
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    def test_ner_output(self):
        processor = TextProcessor(
            dims=64, num_heads=4, vocab_size=100, num_entities=4)
        input_ids = mx.array([[1, 2, 0], [1, 2, 3]])
        pad_token_id = 0
        mask = create_padding_mask(input_ids, pad_token_id)
        _, _, result, _, _ = processor(input_ids, mask)
        expected_shape = (2, 3, 4)  # Batch size 2, seq len 3, 4 entities
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    def test_qa_output(self):
        processor = TextProcessor(dims=64, num_heads=4, vocab_size=100)
        input_ids = mx.array([[1, 2, 0], [1, 2, 3]])
        pad_token_id = 0
        mask = create_padding_mask(input_ids, pad_token_id)
        _, _, _, start_logits, end_logits = processor(input_ids, mask)
        expected_shape = (2, 3)  # Batch size 2, seq len 3
        assert start_logits.shape == expected_shape, f"Expected shape {expected_shape}, got {start_logits.shape}"
        assert end_logits.shape == expected_shape, f"Expected shape {expected_shape}, got {end_logits.shape}"

    def test_sentiment_scalar(self):
        processor = TextProcessor(dims=64, num_heads=4, vocab_size=100)
        input_ids = mx.array([[1, 2, 0], [1, 2, 3]])
        pad_token_id = 0
        mask = create_padding_mask(input_ids, pad_token_id)
        _, label_logits, _, _, _ = processor(input_ids, mask)
        result = mx.sigmoid(label_logits)
        expected_shape = (2, 1)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
        result_scalar = mx.squeeze(result[0]).item()
        assert isinstance(
            result_scalar, float), f"Expected float, got {type(result_scalar)}"
