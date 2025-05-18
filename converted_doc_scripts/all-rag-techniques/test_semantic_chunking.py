import importlib
import numpy as np
import pytest
from helpers import SearchResult
from typing import List
from jet.llm.utils.transformer_embeddings import get_embedding_function

# Dynamically import the module
module = importlib.import_module('2_semantic_chunking')

# Explicitly assign the functions to variables
extract_text_from_chunks = getattr(module, 'extract_text_from_chunks')
split_into_sentences = getattr(module, 'split_into_sentences')
calculate_cosine_similarities = getattr(
    module, 'calculate_cosine_similarities')
compute_breakpoints = getattr(module, 'compute_breakpoints')
split_into_semantic_chunks = getattr(module, 'split_into_semantic_chunks')
perform_semantic_search = getattr(module, 'perform_semantic_search')

# Mock logger for testing


class MockLogger:
    def debug(self, msg): pass
    def info(self, msg): pass
    def success(self, msg, **kwargs): pass


@pytest.fixture
def mock_logger():
    return MockLogger()


@pytest.fixture
def embed_func():
    """Initialize the real embedding function."""
    return get_embedding_function("mxbai-embed-large")


def test_extract_text_from_chunks(mock_logger):
    """Test extracting text from formatted chunks."""
    formatted_chunks = [
        "[doc_index: 0]\n[header: Intro]\n\nThis is an anime about isekai adventures.",
        "[doc_index: 1]\n[header: Plot]\n\nThe hero is transported to a new world."
    ]
    result = extract_text_from_chunks(formatted_chunks, mock_logger)
    expected = "This is an anime about isekai adventures. The hero is transported to a new world."
    assert result == expected


def test_extract_text_from_chunks_empty(mock_logger):
    """Test extracting text from empty chunks."""
    result = extract_text_from_chunks([], mock_logger)
    assert result == ""


def test_split_into_sentences(mock_logger):
    """Test splitting text into sentences."""
    text = "This is a test. Another sentence. Final one."
    result = split_into_sentences(text, mock_logger)
    expected = ["This is a test", "Another sentence", "Final one"]
    assert result == expected


def test_split_into_sentences_single(mock_logger):
    """Test splitting single sentence."""
    text = "This is a test."
    result = split_into_sentences(text, mock_logger)
    expected = ["This is a test"]
    assert result == expected


def test_split_into_sentences_empty(mock_logger):
    """Test splitting empty text."""
    text = ""
    result = split_into_sentences(text, mock_logger)
    expected = []
    assert result == expected


def test_calculate_cosine_similarities(mock_logger):
    """Test cosine similarity calculation between embeddings."""
    embeddings = [
        np.array([1.0, 0.0]),
        np.array([0.707, 0.707]),
        np.array([0.0, 1.0])
    ]
    result = calculate_cosine_similarities(embeddings, mock_logger)
    expected = [0.7071067811865475, 0.7071067811865475]
    assert len(result) == len(embeddings) - 1
    assert all(abs(result[i] - expected[i]) < 1e-6 for i in range(len(result)))


def test_compute_breakpoints_percentile():
    """Test breakpoint computation using percentile method."""
    similarities = [0.9, 0.3, 0.8, 0.2, 0.7]
    result = compute_breakpoints(
        similarities, method="percentile", threshold=30)
    expected = [1, 3]
    assert result == expected


def test_compute_breakpoints_standard_deviation():
    """Test breakpoint computation using standard deviation method."""
    similarities = [0.9, 0.3, 0.8, 0.2, 0.7]
    result = compute_breakpoints(
        similarities, method="standard_deviation", threshold=1)
    mean = np.mean(similarities)
    std_dev = np.std(similarities)
    threshold_value = mean - std_dev
    expected = [i for i, sim in enumerate(
        similarities) if sim < threshold_value]
    assert result == expected


def test_compute_breakpoints_interquartile():
    """Test breakpoint computation using interquartile method."""
    similarities = [0.9, 0.3, 0.8, 0.2, 0.7]
    result = compute_breakpoints(similarities, method="interquartile")
    q1, q3 = np.percentile(similarities, [25, 75])
    threshold_value = q1 - 1.5 * (q3 - q1)
    expected = [i for i, sim in enumerate(
        similarities) if sim < threshold_value]
    assert result == expected


def test_compute_breakpoints_invalid_method():
    """Test error handling for invalid breakpoint method."""
    similarities = [0.9, 0.3, 0.8]
    with pytest.raises(ValueError, match="Invalid method"):
        compute_breakpoints(similarities, method="invalid")


def test_split_into_semantic_chunks(mock_logger):
    """Test splitting sentences into semantic chunks."""
    sentences = ["First sentence", "Second sentence",
                 "Third sentence", "Fourth sentence"]
    breakpoints = [1, 2]
    result = split_into_semantic_chunks(sentences, breakpoints, mock_logger)
    expected = [
        "First sentence. Second sentence.",
        "Third sentence.",
        "Fourth sentence."
    ]
    assert result == expected


def test_split_into_semantic_chunks_no_breakpoints(mock_logger):
    """Test splitting with no breakpoints."""
    sentences = ["First sentence", "Second sentence"]
    breakpoints = []
    result = split_into_semantic_chunks(sentences, breakpoints, mock_logger)
    expected = ["First sentence. Second sentence."]
    assert result == expected


def test_perform_semantic_search(mock_logger, embed_func):
    """Test semantic search with real embedding function."""
    text_chunks = [
        "In this isekai anime, the protagonist is transported to a fantasy world filled with magic.",
        "The hero battles fierce monsters and uncovers ancient secrets in a magical kingdom.",
        "A group of adventurers explores a mysterious dungeon in a parallel world."
    ]
    chunk_embeddings = embed_func(text_chunks)
    query = "isekai anime with magic and adventure"
    result = perform_semantic_search(
        query, text_chunks, chunk_embeddings, embed_func, k=2)
    assert len(result) == 2
    assert all(key in result[0] for key in [
               "id", "rank", "doc_index", "score", "text", "metadata", "relevance_score"])
    assert result[0]['rank'] == 1
    assert result[0]['score'] > result[1]['score']
    assert result[0]['text'] in text_chunks
    assert 0.0 <= result[0]['score'] <= 1.0  # Cosine similarity range


def test_embedding_consistency(embed_func):
    """Test that the embedding function produces consistent embeddings for identical inputs."""
    text = "Isekai anime with a magical world."
    embedding1 = embed_func(text)
    embedding2 = embed_func(text)
    assert len(embedding1) == len(embedding2)
    assert np.allclose(embedding1, embedding2, atol=1e-6)
    assert isinstance(embedding1, list)
    assert all(isinstance(x, float) for x in embedding1)


if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
