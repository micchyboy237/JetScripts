import importlib
import numpy as np
from typing import List, Dict
from jet.logger import CustomLogger
import os
import json
from helpers import generate_ai_response

module = importlib.import_module('3_chunk_size_selector')
ChunkSizeSelector = getattr(module, 'ChunkSizeSelector')

# Test data
TEST_TEXT = (
    "In a world where magic reigns supreme, the kingdom of Eldoria thrives under the rule of King Alaric. "
    "The Great Forest harbors mystical creatures, while the Crystal Mountains hide ancient secrets. "
    "Heroes embark on quests to uncover lost artifacts, facing dragons and dark sorcerers along the way."
)

TEST_QUERY = "Who rules the kingdom of Eldoria?"
TEST_TRUE_ANSWER = "King Alaric rules the kingdom of Eldoria."


def setup_test_logger() -> CustomLogger:
    """Set up a test logger."""
    log_file = "test_chunk_size_selector.log"
    return CustomLogger(log_file, overwrite=True)


def test_extract_text():
    """Test text extraction from formatted chunks."""
    logger = setup_test_logger()
    selector = ChunkSizeSelector(__file__)

    formatted_chunks = [
        "[doc_index: 0]\n[header: Introduction]\n\n" + TEST_TEXT[:50],
        "[doc_index: 1]\n[header: History]\n\n" + TEST_TEXT[50:]
    ]

    result = selector.extract_text(formatted_chunks)
    expected = TEST_TEXT[:50] + " " + TEST_TEXT[50:]
    assert result == expected, f"Expected '{expected}', but got '{result}'"
    logger.info("test_extract_text passed")


def test_chunk_text():
    """Test text chunking with different sizes and overlaps."""
    logger = setup_test_logger()
    selector = ChunkSizeSelector(__file__)

    # Test case 1: Small chunks
    size, overlap = 50, 10
    chunks = selector.chunk_text(TEST_TEXT, size, overlap)
    assert len(chunks) > 0, "Chunks should not be empty"
    assert all(len(chunk) <= size for chunk in chunks), "Chunk size exceeded"

    # Test case 2: Large chunks with no overlap
    size, overlap = 200, 0
    chunks = selector.chunk_text(TEST_TEXT, size, overlap)
    assert len(chunks) == 2, f"Expected 2 chunk, got {len(chunks)}"

    logger.info("test_chunk_text passed")


def test_generate_chunk_dict():
    """Test generation of chunk dictionary for multiple sizes."""
    logger = setup_test_logger()
    selector = ChunkSizeSelector(__file__)

    chunk_dict = selector.generate_chunk_dict(TEST_TEXT)
    assert set(chunk_dict.keys()) == {128, 256, 512}, "Incorrect chunk sizes"
    for size, chunks in chunk_dict.items():
        assert len(chunks) > 0, f"No chunks for size {size}"
        assert all(
            len(chunk) <= size for chunk in chunks), f"Chunk size {size} exceeded"

    logger.info("test_generate_chunk_dict passed")


def test_generate_chunk_embeddings():
    """Test embedding generation for chunks."""
    logger = setup_test_logger()
    selector = ChunkSizeSelector(__file__)
    chunk_dict = {128: [TEST_TEXT[:100], TEST_TEXT[100:200]]}
    embeddings_dict = selector.generate_chunk_embeddings(chunk_dict)
    assert 128 in embeddings_dict, "Expected embeddings for size 128"
    assert len(embeddings_dict[128]) == 2, "Expected 2 embeddings"
    assert isinstance(
        # Changed from np.ndarray to list
        embeddings_dict[128][0], list), "Embedding should be a list"
    assert all(isinstance(x, float)
               for x in embeddings_dict[128][0]), "Embedding elements should be floats"
    logger.info("test_generate_chunk_embeddings passed")


def test_retrieve_relevant_chunks():
    """Test retrieval of relevant chunks for a query."""
    logger = setup_test_logger()
    selector = ChunkSizeSelector(__file__)

    text_chunks = [TEST_TEXT[:100], TEST_TEXT[100:200], TEST_TEXT[200:]]
    embeddings = selector.embed_func(text_chunks)

    results = selector.retrieve_relevant_chunks(
        TEST_QUERY, text_chunks, embeddings, k=2)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert all(isinstance(r["score"], float)
               for r in results), "Scores should be floats"
    assert all(
        r["text"] in text_chunks for r in results), "Retrieved texts should match input"

    logger.info("test_retrieve_relevant_chunks passed")


def test_evaluate_response():
    """Test response evaluation for faithfulness and relevancy."""
    logger = setup_test_logger()
    selector = ChunkSizeSelector(__file__)

    # Test case 1: Perfect response
    response = TEST_TRUE_ANSWER
    faith_score, rel_score = selector.evaluate_response(
        TEST_QUERY, response, TEST_TRUE_ANSWER)
    assert faith_score == 1.0, f"Expected faithfulness 1.0, got {faith_score}"
    assert rel_score == 1.0, f"Expected relevancy 1.0, got {rel_score}"

    # Test case 2: Partial response
    response = "King Alaric is a ruler in Eldoria."
    faith_score, rel_score = selector.evaluate_response(
        TEST_QUERY, response, TEST_TRUE_ANSWER)
    assert faith_score >= 0.5, f"Expected faithfulness >= 0.5, got {faith_score}"
    assert rel_score >= 0.5, f"Expected relevancy >= 0.5, got {rel_score}"

    # Test case 3: Incorrect response
    response = "Queen Elizabeth rules Eldoria."
    faith_score, rel_score = selector.evaluate_response(
        TEST_QUERY, response, TEST_TRUE_ANSWER)
    assert faith_score == 0.0, f"Expected faithfulness 0.0, got {faith_score}"
    assert rel_score == 0.0, f"Expected relevancy 0.0, got {rel_score}"

    logger.info("test_evaluate_response passed")


def test_integration():
    """Test full pipeline integration."""
    logger = setup_test_logger()
    selector = ChunkSizeSelector(__file__)

    # Generate chunks
    chunk_dict = selector.generate_chunk_dict(TEST_TEXT)

    # Generate embeddings
    embeddings_dict = selector.generate_chunk_embeddings(chunk_dict)

    # Retrieve chunks
    retrieved_chunks = selector.retrieve_relevant_chunks(
        TEST_QUERY,
        chunk_dict[256],
        embeddings_dict[256]
    )

    # Generate response
    system_prompt = (
        "You are an AI assistant that strictly answers based on the given context. "
        "If the answer cannot be derived directly from the provided context, "
        "respond with: 'I do not have enough information to answer that.'"
    )
    response = generate_ai_response(
        TEST_QUERY,
        system_prompt,
        retrieved_chunks,
        selector.mlx,
        logger
    )

    # Evaluate response
    faith_score, rel_score = selector.evaluate_response(
        TEST_QUERY,
        response,
        TEST_TRUE_ANSWER
    )

    assert response, "Response should not be empty"
    assert faith_score >= 0.0 and rel_score >= 0.0, "Scores should be non-negative"

    logger.info("test_integration passed")


if __name__ == "__main__":
    test_extract_text()
    test_chunk_text()
    test_generate_chunk_dict()
    test_generate_chunk_embeddings()
    test_retrieve_relevant_chunks()
    test_evaluate_response()
    test_integration()
    print("All tests passed!")
