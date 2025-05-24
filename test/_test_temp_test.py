from jet.token.token_utils import split_headers
import pytest
from unittest.mock import Mock, patch
from jet.vectors.document_types import HeaderDocument, HeaderTextNode
from jet.utils.doc_utils import add_parent_child_relationship, add_sibling_relationship
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship
from typing import List, Optional, Callable
from uuid import uuid4

# Mock SentenceSplitter class


class MockSentenceSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_texts = []

    def split_text(self, text: str) -> List[str]:
        return self.split_texts

# Test fixture for HeaderDocument


@pytest.fixture
def header_doc():
    return HeaderDocument(
        text="This is a test document.",
        doc_index=1,
        header_level=1,
        header="Test Header",
        parent_header="Parent Header",
        content="This is a test document."
    )

# Test cases


def test_split_headers_single_doc_below_max_tokens(header_doc):
    """Test that documents with token count <= effective_max_tokens are not split."""
    tokens = [[1, 2, 3]]  # 3 tokens
    chunk_size = 5
    buffer = 1
    effective_max_tokens = chunk_size - buffer

    with patch('jet.llm.embeddings.sentence_embedding.get_tokenizer_fn') as mock_tokenizer, \
            patch('jet.llm.mlx.models.get_embedding_size', return_value=chunk_size):
        mock_tokenizer.return_value = lambda x: [[1, 2, 3]]

        nodes = split_headers(
            docs=header_doc,
            tokens=tokens,
            chunk_size=chunk_size,
            chunk_overlap=0,
            buffer=buffer
        )

    assert len(nodes) == 1
    assert nodes[0].text == header_doc.text
    assert nodes[0].metadata["chunk_index"] is None
    assert nodes[0].metadata["start_idx"] == 0
    assert nodes[0].metadata["end_idx"] == len(header_doc.text)
    assert nodes[0].metadata["content"] == header_doc.text
    assert nodes[0].metadata["header"] == "Test Header"
    assert nodes[0].metadata["parent_header"] == "Parent Header"


def test_split_headers_single_doc_exceeds_max_tokens(header_doc):
    """Test that documents with token count > effective_max_tokens are split correctly."""
    tokens = [[1, 2, 3, 4, 5, 6]]  # 6 tokens
    chunk_size = 4
    buffer = 1
    chunk_overlap = 1
    effective_max_tokens = chunk_size - buffer

    # Mock SentenceSplitter to return specific chunks
    mock_splitter = MockSentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    mock_splitter.split_texts = ["This is a test", "test document."]

    with patch('llama_index.core.node_parser.SentenceSplitter', return_value=mock_splitter), \
            patch('jet.utils.doc_utils.add_parent_child_relationship', side_effect=add_parent_child_relationship), \
            patch('jet.utils.doc_utils.add_sibling_relationship', side_effect=add_sibling_relationship), \
            patch('jet.llm.mlx.models.get_embedding_size', return_value=chunk_size), \
            patch('jet.llm.embeddings.sentence_embedding.get_tokenizer_fn') as mock_tokenizer:
        mock_tokenizer.return_value = lambda x: tokens

        nodes = split_headers(
            docs=header_doc,
            tokens=tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            buffer=buffer
        )

    assert len(nodes) == 2
    assert nodes[0].text == "This is a test"
    assert nodes[0].metadata["chunk_index"] == 0
    assert nodes[0].metadata["start_idx"] == 0
    assert nodes[0].metadata["end_idx"] == 14
    assert nodes[0].metadata["content"] == "This is a test"
    assert nodes[0].metadata["header"] == "Test Header"
    assert nodes[0].metadata["parent_header"] == "Parent Header"

    assert nodes[1].text == "test document."
    assert nodes[1].metadata["chunk_index"] == 1
    assert nodes[1].metadata["start_idx"] == 10
    assert nodes[1].metadata["end_idx"] == 24
    assert nodes[1].metadata["content"] == "test document."
    assert nodes[1].metadata["header"] == "Test Header"
    assert nodes[1].metadata["parent_header"] == "Parent Header"

    # Verify relationships
    assert nodes[0].parent_node is not None
    assert nodes[1].parent_node == nodes[0].parent_node
    assert nodes[0].relationships[NodeRelationship.NEXT].node_id == nodes[1].id_
    assert nodes[0].relationships[NodeRelationship.NEXT].metadata["content"] == "test document."
    assert nodes[1].relationships[NodeRelationship.PREVIOUS].node_id == nodes[0].id_
    assert nodes[1].relationships[NodeRelationship.PREVIOUS].metadata["content"] == "This is a test"


def test_split_headers_multiple_docs_mixed_tokens(header_doc):
    """Test handling of multiple documents, some exceeding and some below effective_max_tokens."""
    doc2 = HeaderDocument(
        text="Short text.",
        doc_index=2,
        header_level=1,
        header="Test Header 2",
        parent_header="Parent Header 2",
        content="Short text."
    )
    # First doc: 6 tokens, Second doc: 2 tokens
    tokens = [[1, 2, 3, 4, 5, 6], [1, 2]]
    chunk_size = 4
    buffer = 1
    chunk_overlap = 1
    effective_max_tokens = chunk_size - buffer

    mock_splitter = MockSentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    mock_splitter.split_texts = ["This is a test", "test document."]

    with patch('llama_index.core.node_parser.SentenceSplitter', return_value=mock_splitter), \
            patch('jet.utils.doc_utils.add_parent_child_relationship', side_effect=add_parent_child_relationship), \
            patch('jet.utils.doc_utils.add_sibling_relationship', side_effect=add_sibling_relationship), \
            patch('jet.llm.mlx.models.get_embedding_size', return_value=chunk_size), \
            patch('jet.llm.embeddings.sentence_embedding.get_tokenizer_fn') as mock_tokenizer:
        mock_tokenizer.return_value = lambda x: tokens

        nodes = split_headers(
            docs=[header_doc, doc2],
            tokens=tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            buffer=buffer
        )

    assert len(nodes) == 3
    # First document (split into two chunks)
    assert nodes[0].text == "This is a test"
    assert nodes[0].metadata["doc_index"] == 1
    assert nodes[0].metadata["chunk_index"] == 0
    assert nodes[0].metadata["start_idx"] == 0
    assert nodes[0].metadata["end_idx"] == 14
    assert nodes[0].metadata["header"] == "Test Header"
    assert nodes[0].metadata["parent_header"] == "Parent Header"
    assert nodes[1].text == "test document."
    assert nodes[1].metadata["doc_index"] == 1
    assert nodes[1].metadata["chunk_index"] == 1
    assert nodes[1].metadata["start_idx"] == 10
    assert nodes[1].metadata["end_idx"] == 24
    assert nodes[1].metadata["header"] == "Test Header"
    assert nodes[1].metadata["parent_header"] == "Parent Header"
    # Second document (not split)
    assert nodes[2].text == "Short text."
    assert nodes[2].metadata["doc_index"] == 2
    assert nodes[2].metadata["chunk_index"] is None
    assert nodes[2].metadata["header"] == "Test Header 2"
    assert nodes[2].metadata["parent_header"] == "Parent Header 2"
    # Verify sibling relationships
    assert nodes[0].relationships[NodeRelationship.NEXT].node_id == nodes[1].id_
    assert nodes[0].relationships[NodeRelationship.NEXT].metadata["content"] == "test document."
    assert nodes[1].relationships[NodeRelationship.PREVIOUS].node_id == nodes[0].id_
    assert nodes[1].relationships[NodeRelationship.PREVIOUS].metadata["content"] == "This is a test"


def test_split_headers_invalid_chunk_size(header_doc):
    """Test that ValueError is raised when chunk_size <= chunk_overlap."""
    with pytest.raises(ValueError, match="Chunk size.*must be greater than chunk overlap"):
        split_headers(
            docs=header_doc,
            chunk_size=5,
            chunk_overlap=5
        )


def test_split_headers_invalid_effective_max_tokens(header_doc):
    """Test that ValueError is raised when effective_max_tokens <= chunk_overlap."""
    with pytest.raises(ValueError, match="Effective max tokens.*must be greater than chunk_overlap"):
        split_headers(
            docs=header_doc,
            chunk_size=5,
            chunk_overlap=4,
            buffer=2
        )


def test_split_headers_mismatched_tokens_length(header_doc):
    """Test that ValueError is raised when tokens length doesn't match documents length."""
    tokens = [[1, 2], [3, 4]]  # Two token lists, but only one document
    with pytest.raises(ValueError, match="Length of provided tokens.*does not match number of documents"):
        split_headers(
            docs=header_doc,
            tokens=tokens,
            chunk_size=5
        )
