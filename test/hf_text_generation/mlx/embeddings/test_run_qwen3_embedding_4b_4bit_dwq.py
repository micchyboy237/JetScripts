import pytest
import numpy as np
import mlx.core as mx
from unittest.mock import Mock
from transformers import AutoTokenizer

# Assuming the code to be tested is in a module named `retrieval`
from run_qwen3_embedding_4b_4bit_dwq import format_instruction, process_inputs, compute_logits


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ", padding_side='left'
    )
    tokenizer.encode = Mock(wraps=tokenizer.encode)
    tokenizer.pad = Mock(wraps=tokenizer.pad)
    return tokenizer


@pytest.fixture
def mock_model():
    model = Mock()
    # Simulate model output: [batch_size, seq_len, vocab_size]
    model.return_value = mx.array(
        np.random.randn(2, 1024, 32000), dtype=mx.float32)
    return model


@pytest.fixture
def max_length():
    return 1024


@pytest.fixture
def prefix_tokens(tokenizer):
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    return list(tokenizer.encode(prefix, padding=True, add_special_tokens=False))


@pytest.fixture
def suffix_tokens(tokenizer):
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return list(tokenizer.encode(suffix, padding=True, add_special_tokens=False))


@pytest.fixture
def token_ids(tokenizer):
    return {
        'true_id': tokenizer.convert_tokens_to_ids("yes"),
        'false_id': tokenizer.convert_tokens_to_ids("no")
    }


class TestFormatInstruction:
    def test_default_instruction(self):
        instruction = None
        query = "What is the capital of China?"
        doc = "The capital of China is Beijing."
        expected = "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: What is the capital of China?\n<Document>: The capital of China is Beijing."
        result = format_instruction(instruction, query, doc)
        assert result == expected

    def test_custom_instruction(self):
        instruction = "Evaluate the relevance of the document."
        query = "Explain gravity"
        doc = "Gravity is a force."
        expected = "<Instruct>: Evaluate the relevance of the document.\n<Query>: Explain gravity\n<Document>: Gravity is a force."
        result = format_instruction(instruction, query, doc)
        assert result == expected

    def test_empty_query(self):
        instruction = "Test instruction"
        query = ""
        doc = "Some document."
        expected = "<Instruct>: Test instruction\n<Query>: \n<Document>: Some document."
        result = format_instruction(instruction, query, doc)
        assert result == expected


class TestProcessInputs:
    def test_single_pair(self, tokenizer, max_length, prefix_tokens, suffix_tokens):
        pairs = ["<Instruct>: Test\n<Query>: Query\n<Document>: Doc"]
        expected_input_ids = np.array([prefix_tokens + tokenizer.encode(pairs[0], add_special_tokens=False)[
                                      :max_length - len(prefix_tokens) - len(suffix_tokens)] + suffix_tokens])
        expected_attention_mask = np.ones((1, max_length), dtype=np.int32)

        tokenizer.pad.return_value = {
            'input_ids': expected_input_ids,
            'attention_mask': expected_attention_mask
        }

        result = process_inputs(
            pairs, tokenizer, max_length, prefix_tokens, suffix_tokens)

        assert isinstance(result['input_ids'], mx.array)
        assert isinstance(result['attention_mask'], mx.array)
        assert result['input_ids'].dtype == mx.int32
        assert result['attention_mask'].dtype == mx.int32
        assert result['input_ids'].shape == (1, max_length)
        assert result['attention_mask'].shape == (1, max_length)

    def test_multiple_pairs(self, tokenizer, max_length, prefix_tokens, suffix_tokens):
        pairs = [
            "<Instruct>: Test\n<Query>: Query1\n<Document>: Doc1",
            "<Instruct>: Test\n<Query>: Query2\n<Document>: Doc2"
        ]
        encoded = [tokenizer.encode(p, add_special_tokens=False)[
            :max_length - len(prefix_tokens) - len(suffix_tokens)] for p in pairs]
        expected_input_ids = np.array([
            prefix_tokens + encoded[0] + suffix_tokens,
            prefix_tokens + encoded[1] + suffix_tokens
        ])
        expected_attention_mask = np.ones((2, max_length), dtype=np.int32)

        tokenizer.pad.return_value = {
            'input_ids': expected_input_ids,
            'attention_mask': expected_attention_mask
        }

        result = process_inputs(
            pairs, tokenizer, max_length, prefix_tokens, suffix_tokens)

        assert result['input_ids'].shape == (2, max_length)
        assert result['attention_mask'].shape == (2, max_length)

    def test_truncation(self, tokenizer, max_length, prefix_tokens, suffix_tokens):
        long_pair = "<Instruct>: Test\n<Query>: " + "x" * 2000 + "\n<Document>: Doc"
        max_input_len = max_length - len(prefix_tokens) - len(suffix_tokens)
        encoded = tokenizer.encode(long_pair, add_special_tokens=False)[
            :max_input_len]
        expected_input_ids = np.array(
            [prefix_tokens + encoded + suffix_tokens])
        expected_attention_mask = np.ones((1, max_length), dtype=np.int32)

        tokenizer.pad.return_value = {
            'input_ids': expected_input_ids,
            'attention_mask': expected_attention_mask
        }

        result = process_inputs([long_pair], tokenizer,
                                max_length, prefix_tokens, suffix_tokens)

        assert result['input_ids'].shape[1] == max_length
        assert mx.sum(result['attention_mask']) == max_length


class TestComputeLogits:
    def test_compute_logits_output(self, mock_model, token_ids, max_length):
        inputs = {
            'input_ids': mx.array(np.zeros((2, max_length), dtype=np.int32)),
            'attention_mask': mx.array(np.ones((2, max_length), dtype=np.int32))
        }
        result = compute_logits(
            mock_model, inputs, token_ids['true_id'], token_ids['false_id'])

        expected = [float(mx.softmax(mx.array(np.random.randn(2)), axis=1)[
                          :, 1][i]) for i in range(2)]
        assert len(result) == 2
        assert all(isinstance(score, float) for score in result)
        assert all(0 <= score <= 1 for score in result)

    def test_empty_input(self, mock_model, token_ids, max_length):
        inputs = {
            'input_ids': mx.array(np.zeros((0, max_length), dtype=np.int32)),
            'attention_mask': mx.array(np.zeros((0, max_length), dtype=np.int32))
        }
        mock_model.return_value = mx.array(
            np.random.randn(0, max_length, 32000), dtype=mx.float32)
        result = compute_logits(
            mock_model, inputs, token_ids['true_id'], token_ids['false_id'])

        expected = []
        assert result == expected

    def test_single_input(self, mock_model, token_ids, max_length):
        inputs = {
            'input_ids': mx.array(np.zeros((1, max_length), dtype=np.int32)),
            'attention_mask': mx.array(np.ones((1, max_length), dtype=np.int32))
        }
        result = compute_logits(
            mock_model, inputs, token_ids['true_id'], token_ids['false_id'])

        expected = [
            float(mx.softmax(mx.array(np.random.randn(2)), axis=1)[:, 1][0])]
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert 0 <= result[0] <= 1
