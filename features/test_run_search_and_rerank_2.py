import pytest
from typing import List, Dict
from collections import defaultdict
from unittest.mock import Mock
from jet.models.model_types import LLMModelType

from run_search_and_rerank_2 import group_results_by_url_for_llm_context

# Mock tokenizer function


def mock_tokenizer(text: str) -> List[int]:
    return list(range(len(text.split())))


@pytest.fixture
def mock_get_tokenizer_fn():
    return Mock(return_value=mock_tokenizer)


@pytest.fixture
def sample_documents():
    return [
        {
            "text": "Content for introduction",
            "url": "https://example.com",
            "parent_header": "## Introduction",
            "header": "## Sub Introduction",
            "level": 2,
            "parent_level": 2,
            "num_tokens": 3,
            "score": 0.9,
            "merged_doc_id": "doc1",
            "chunk_id": "chunk1",
            "doc_index": 0,
            "chunk_index": 0,
        },
        {
            "text": "Duplicate introduction content",
            "url": "https://example.com",
            "parent_header": "## Introduction",
            "header": "## Introduction",
            "level": 2,
            "parent_level": 2,
            "num_tokens": 3,
            "score": 0.85,
            "merged_doc_id": "doc2",
            "chunk_id": "chunk2",
            "doc_index": 1,
            "chunk_index": 1,
        },
        {
            "text": "Content for another section",
            "url": "https://example.com",
            "parent_header": "# Main Section",
            "header": "## Another Section",
            "level": 2,
            "parent_level": 1,
            "num_tokens": 4,
            "score": 0.8,
            "merged_doc_id": "doc3",
            "chunk_id": "chunk3",
            "doc_index": 2,
            "chunk_index": 2,
        },
        {
            "text": "Content for another URL",
            "url": "https://another.com",
            "parent_header": "## Introduction",
            "header": None,
            "level": 0,
            "parent_level": 2,
            "num_tokens": 4,
            "score": 0.7,
            "merged_doc_id": "doc4",
            "chunk_id": "chunk4",
            "doc_index": 3,
            "chunk_index": 0,
        },
    ]


class TestGroupResultsByUrlForLLMContext:
    def test_deduplicates_identical_parent_and_child_headers(self, mock_get_tokenizer_fn, sample_documents):
        # Given: Documents with identical parent_header and header text
        documents = sample_documents
        llm_model = "mock-model"  # type: LLMModelType
        max_tokens = 1000
        buffer = 100
        expected = (
            "<!-- Source: https://example.com -->\n\n"
            "## Introduction\n\n"
            "Content for introduction\n\n"
            "# Main Section\n\n"
            "## Another Section\n\n"
            "Content for another section\n\n"
            "<!-- Source: https://another.com -->\n\n"
            "## Introduction\n\n"
            "Content for another URL"
        )

        # When: The function processes the documents
        result = group_results_by_url_for_llm_context(
            documents, llm_model, max_tokens, buffer)

        # Then: Duplicate headers are skipped, keeping the highest-scoring document
        assert result.strip() == expected.strip()
        assert "Duplicate introduction content" not in result
        # Skipped due to duplicate with parent_header
        assert "## Sub Introduction" not in result

    def test_deduplicates_headers_across_parent_groups(self, mock_get_tokenizer_fn):
        # Given: Documents with the same header text in different parent groups
        documents = [
            {
                "text": "Content for introduction under main",
                "url": "https://example.com",
                "parent_header": "# Main Section",
                "header": "## Introduction",
                "level": 2,
                "parent_level": 1,
                "num_tokens": 5,
                "score": 0.9,
                "merged_doc_id": "doc5",
                "chunk_id": "chunk5",
                "doc_index": 4,
                "chunk_index": 0,
            },
            {
                "text": "Duplicate introduction content",
                "url": "https://example.com",
                "parent_header": "# Another Section",
                "header": "## Introduction",
                "level": 2,
                "parent_level": 1,
                "num_tokens": 4,
                "score": 0.85,
                "merged_doc_id": "doc6",
                "chunk_id": "chunk6",
                "doc_index": 5,
                "chunk_index": 1,
            },
        ]
        llm_model = "mock-model"  # type: LLMModelType
        max_tokens = 1000
        buffer = 100
        expected = (
            "<!-- Source: https://example.com -->\n\n"
            "# Main Section\n\n"
            "## Introduction\n\n"
            "Content for introduction under main"
        )

        # When: The function processes documents with duplicate headers across parent groups
        result = group_results_by_url_for_llm_context(
            documents, llm_model, max_tokens, buffer)

        # Then: Only the highest-scoring document with the header is included
        assert result.strip() == expected.strip()
        assert "Duplicate introduction content" not in result

    def test_handles_empty_parent_header(self, mock_get_tokenizer_fn):
        # Given: Documents with no parent header
        documents = [
            {
                "text": "Content without parent",
                "url": "https://example.com",
                "parent_header": "None",
                "header": None,
                "level": 0,
                "parent_level": None,
                "num_tokens": 3,
                "score": 0.9,
                "merged_doc_id": "doc7",
                "chunk_id": "chunk7",
                "doc_index": 6,
                "chunk_index": 0,
            }
        ]
        llm_model = "mock-model"  # type: LLMModelType
        max_tokens = 1000
        buffer = 100
        expected = (
            "<!-- Source: https://example.com -->\n\n"
            "Content without parent"
        )

        # When: The function processes documents without parent headers
        result = group_results_by_url_for_llm_context(
            documents, llm_model, max_tokens, buffer)

        # Then: The output includes content without unnecessary headers
        assert result.strip() == expected.strip()

    def test_respects_token_limit(self, mock_get_tokenizer_fn, sample_documents):
        # Given: Documents that exceed token limit
        documents = sample_documents
        llm_model = "mock-model"  # type: LLMModelType
        max_tokens = 20  # Low limit to force truncation
        buffer = 5
        expected = (
            "<!-- Source: https://example.com -->\n\n"
            "## Introduction\n\n"
            "Content for introduction"
        )

        # When: The function processes documents with a strict token limit
        result = group_results_by_url_for_llm_context(
            documents, llm_model, max_tokens, buffer)

        # Then: Only documents within token limit are included
        assert result.strip() == expected.strip()
        assert "Content for another section" not in result
        assert "Content for another URL" not in result

    def test_handles_identical_content_different_headers(self, mock_get_tokenizer_fn):
        # Given: Documents with identical content but different headers
        documents = [
            {
                "text": "Identical content",
                "url": "https://example.com",
                "parent_header": "# Main Section",
                "header": "## Header A",
                "level": 2,
                "parent_level": 1,
                "num_tokens": 3,
                "score": 0.9,
                "merged_doc_id": "doc8",
                "chunk_id": "chunk8",
                "doc_index": 7,
                "chunk_index": 0,
            },
            {
                "text": "Identical content",
                "url": "https://example.com",
                "parent_header": "# Main Section",
                "header": "## Header B",
                "level": 2,
                "parent_level": 1,
                "num_tokens": 3,
                "score": 0.85,
                "merged_doc_id": "doc9",
                "chunk_id": "chunk9",
                "doc_index": 8,
                "chunk_index": 1,
            },
        ]
        llm_model = "mock-model"  # type: LLMModelType
        max_tokens = 1000
        buffer = 100
        expected = (
            "<!-- Source: https://example.com -->\n\n"
            "# Main Section\n\n"
            "## Header A\n\n"
            "Identical content\n\n"
            "## Header B\n\n"
            "Identical content"
        )

        # When: The function processes documents with identical content
        result = group_results_by_url_for_llm_context(
            documents, llm_model, max_tokens, buffer)

        # Then: Both headers are included since they are different
        assert result.strip() == expected.strip()
