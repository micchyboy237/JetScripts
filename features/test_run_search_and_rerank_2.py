import unittest
from typing import List, Dict
from collections import defaultdict
import os
from unittest.mock import Mock, patch

# Assuming these are defined in your module
from run_search_and_rerank_2 import group_results_by_url_for_llm_context, LLMModelType

# Mock logger to capture debug/warning messages
logger = Mock()

# Mock tokenizer class to mimic a tokenizer with an encode method


class MockTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False, remove_pad_tokens: bool = True) -> List[int]:
        # Simple tokenization: count words as tokens, plus 2 for newlines
        return list(range(len(text.split()) + (2 if '\n\n' in text else 0)))

# Mock save_file function


def mock_save_file(data: Dict, path: str):
    pass


class TestGroupResultsByUrlForLlmContext(unittest.TestCase):
    def setUp(self):
        self.max_tokens = 2000
        self.buffer = 100
        self.llm_model = Mock(spec=LLMModelType)
        # Mock the tokenizer function to return a MockTokenizer instance
        self.patcher = patch(
            'run_search_and_rerank_2.get_tokenizer_fn', return_value=MockTokenizer())
        self.patcher.start()
        # Mock save_file
        self.save_file_patcher = patch(
            'run_search_and_rerank_2.save_file', side_effect=mock_save_file)
        self.save_file_patcher.start()
        # Mock OUTPUT_DIR
        self.output_dir_patcher = patch(
            'run_search_and_rerank_2.OUTPUT_DIR', '/tmp')
        self.output_dir_patcher.start()
        # Mock logger
        self.logger_patcher = patch('run_search_and_rerank_2.logger', logger)
        self.logger_patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.save_file_patcher.stop()
        self.output_dir_patcher.stop()
        self.logger_patcher.stop()

    def test_duplicate_parent_and_child_header(self):
        """Test case where parent header matches a child header's text."""
        documents = [
            {
                "text": "Fall 2025 has its villainess isekai anime, but this one comes with a clever twist.",
                "url": "https://example.com",
                "parent_header": "### A Wild Last Boss Appeared!",
                "header": "#### The Dark History of the Reincarnated Villainess",
                "level": 4,
                "parent_level": 3,
                "score": 0.9,
                "num_tokens": 20,
                "chunk_index": 0
            },
            {
                "text": "Fans of OP isekai anime protagonists will be eating well throughout 2025, and A Wild Last Boss Appeared should be a particularly great feast.",
                "url": "https://example.com",
                "parent_header": "### A Wild Last Boss Appeared!",
                "header": "#### A Wild Last Boss Appeared!",
                "level": 4,
                "parent_level": 3,
                "score": 0.8,
                "num_tokens": 30,
                "chunk_index": 1
            }
        ]
        expected_output = (
            "<!-- Source: https://example.com -->\n\n"
            "#### The Dark History of the Reincarnated Villainess\n\n"
            "Fall 2025 has its villainess isekai anime, but this one comes with a clever twist.\n\n"
            "#### A Wild Last Boss Appeared!\n\n"
            "Fans of OP isekai anime protagonists will be eating well throughout 2025, and A Wild Last Boss Appeared should be a particularly great feast."
        )
        result = group_results_by_url_for_llm_context(
            documents, self.llm_model, self.max_tokens, self.buffer)
        print("Actual output:\n", result)  # Print actual output for debugging
        self.assertEqual(result.strip(), expected_output.strip())

    def test_non_duplicate_headers(self):
        """Test case where parent and child headers are different."""
        documents = [
            {
                "text": "This is a unique section about isekai trends.",
                "url": "https://example.com",
                "parent_header": "### Isekai Trends",
                "header": "#### New Releases",
                "level": 4,
                "parent_level": 3,
                "score": 0.7,
                "num_tokens": 10,
                "chunk_index": 0
            }
        ]
        expected_output = (
            "<!-- Source: https://example.com -->\n\n"
            "### Isekai Trends\n\n"
            "#### New Releases\n\n"
            "This is a unique section about isekai trends."
        )
        result = group_results_by_url_for_llm_context(
            documents, self.llm_model, self.max_tokens, self.buffer)
        self.assertEqual(result.strip(), expected_output.strip())

    def test_multiple_child_headers(self):
        """Test case with multiple child headers, one matching parent."""
        documents = [
            {
                "text": "First section text.",
                "url": "https://example.com",
                "parent_header": "### Main Topic",
                "header": "#### Subtopic One",
                "level": 4,
                "parent_level": 3,
                "score": 0.9,
                "num_tokens": 10,
                "chunk_index": 0
            },
            {
                "text": "Second section text.",
                "url": "https://example.com",
                "parent_header": "### Main Topic",
                "header": "#### Main Topic",
                "level": 4,
                "parent_level": 3,
                "score": 0.8,
                "num_tokens": 10,
                "chunk_index": 1
            }
        ]
        expected_output = (
            "<!-- Source: https://example.com -->\n\n"
            "#### Subtopic One\n\n"
            "First section text.\n\n"
            "#### Main Topic\n\n"
            "Second section text."
        )
        result = group_results_by_url_for_llm_context(
            documents, self.llm_model, self.max_tokens, self.buffer)
        self.assertEqual(result.strip(), expected_output.strip())

    def test_no_parent_header(self):
        """Test case with no parent header."""
        documents = [
            {
                "text": "Content without a parent header.",
                "url": "https://example.com",
                "parent_header": "None",
                "header": "#### Standalone Section",
                "level": 4,
                "parent_level": None,
                "score": 0.6,
                "num_tokens": 10,
                "chunk_index": 0
            }
        ]
        expected_output = (
            "<!-- Source: https://example.com -->\n\n"
            "#### Standalone Section\n\n"
            "Content without a parent header."
        )
        result = group_results_by_url_for_llm_context(
            documents, self.llm_model, self.max_tokens, self.buffer)
        self.assertEqual(result.strip(), expected_output.strip())

    def test_token_limit_exceeded(self):
        """Test case where documents exceed token limit."""
        documents = [
            {
                "text": "This is a very long text " * 1000,  # Exceeds max_tokens
                "url": "https://example.com",
                "parent_header": "### Main Topic",
                "header": "#### Subtopic",
                "level": 4,
                "parent_level": 3,
                "score": 0.9,
                "num_tokens": 2001,
                "chunk_index": 0
            }
        ]
        result = group_results_by_url_for_llm_context(
            documents, self.llm_model, self.max_tokens, self.buffer)
        self.assertEqual(result, "")  # Expect empty output due to token limit
        logger.debug.assert_called_with(
            "Skipping document (score: 0.9): would exceed max_tokens (2001 > 1900)")


if __name__ == '__main__':
    unittest.main()
