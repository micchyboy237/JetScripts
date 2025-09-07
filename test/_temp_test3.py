import pytest
from typing import List
from pathlib import Path
import json
from datetime import datetime
from unittest.mock import Mock
import numpy as np

# Import classes and functions from main code
from contextual_chunk_headers_rag import PersistentMemoryStore, VectorStore, extract_headers_and_text, chunk_text_with_headers, generate_embeddings, process_query_with_iterative_memory, QueryResult


@pytest.fixture
def memory_store(tmp_path: Path):
    """Fixture to create a fresh PersistentMemoryStore instance."""
    db_path = tmp_path / "test_rag_memory.db"
    store = PersistentMemoryStore(db_path=str(db_path))
    yield store
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def mock_openai_client():
    """Fixture to create a mock OpenAI client."""
    client = Mock()
    client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1, 0.2, 0.3])])
    client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Mock response"))])
    return client


def test_extract_headers_and_text():
    """Test header and text extraction from web content."""
    # Given: Sample web content with headers
    web_content = """
    <h1>Main Header</h1>
    <p>This is the main content.</p>
    <h2>Sub Header</h2>
    <p>This is sub content.</p>
    """
    expected = [
        {"header": "Main Header", "text": "This is the main content."},
        {"header": "Sub Header", "text": "This is sub content."}
    ]

    # When: Extracting headers and text
    result = extract_headers_and_text(web_content)

    # Then: Headers and text should match expected
    assert result == expected, "Extracted chunks do not match expected"


def test_chunk_text_with_headers():
    """Test chunking text while preserving headers."""
    # Given: Sample chunks with headers
    chunks = [{"header": "Test Header",
               "text": "This is a long text that needs to be chunked into smaller pieces for processing."}]
    expected = [
        {"header": "Test Header",
            "text": "This is a long text that needs to be chunked into smaller pieces for processing."}
    ]

    # When: Chunking text with headers
    result = chunk_text_with_headers(chunks, chunk_size=50, overlap=10)

    # Then: Chunks should preserve headers and respect size
    assert len(result) >= 1, "Expected at least one chunk"
    assert all(chunk["header"] ==
               "Test Header" for chunk in result), "Headers do not match"
    assert all(len(chunk["text"].split()) <=
               50 for chunk in result), "Chunk size exceeded"


def test_vector_store_search():
    """Test vector store search functionality."""
    # Given: A vector store with sample chunks and embeddings
    vector_store = VectorStore()
    chunks = [
        {"header": "Header1", "text": "Content1"},
        {"header": "Header2", "text": "Content2"}
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    vector_store.add_chunks(chunks, embeddings)
    query_embedding = [1.0, 0.0]
    expected = [{"header": "Header1", "text": "Content1"}]

    # When: Searching for top_k=1
    result = vector_store.search(query_embedding, top_k=1)

    # Then: Most similar chunk should be returned
    assert result == expected, "Search result does not match expected"


def test_persistent_memory_store_add_and_retrieve(memory_store: PersistentMemoryStore):
    """Test adding and retrieving results from PersistentMemoryStore."""
    # Given: A query, chunks, response, and timestamp
    query = "Test query"
    expected_chunks = ["Chunk1", "Chunk2"]
    expected_response = "Test response"
    timestamp = "2025-09-07T19:19:00"

    # When: Adding a result to the memory store
    memory_store.add_result(query, expected_chunks,
                            expected_response, timestamp)

    # Then: The result should be retrievable
    result = memory_store.get_consolidated_results(query)
    assert len(result) == 1, "Expected one result"
    assert result[0]["query"] == query, "Query does not match"
    assert result[0]["retrieved_chunks"] == expected_chunks, "Chunks do not match"
    assert result[0]["response"] == expected_response, "Response does not match"
    assert result[0]["timestamp"] == timestamp, "Timestamp does not match"


def test_process_query_with_iterative_memory(mock_openai_client, memory_store: PersistentMemoryStore):
    """Test the full query processing pipeline with iterative memory."""
    # Given: Sample web content, query, and mock client
    web_content = """
    <h1>About France</h1>
    <p>France is a country in Western Europe. Its capital is Paris.</p>
    """
    query = "What is the capital of France?"
    timestamp = "2025-09-07T19:19:00"
    expected_response = "Mock response"

    # When: Processing the query
    response = process_query_with_iterative_memory(
        query, web_content, mock_openai_client, memory_store, timestamp, max_iterations=1
    )

    # Then: Response should match mock and be stored in memory
    assert response == expected_response, "Response does not match expected"
    results = memory_store.get_consolidated_results(query)
    assert len(results) == 1, "Expected one result in memory"
    assert results[0]["response"] == expected_response, "Stored response does not match"


@pytest.fixture(autouse=True)
def cleanup_memory_store(memory_store: PersistentMemoryStore):
    """Ensure memory store is cleaned up after each test."""
    yield
    with sqlite3.connect(memory_store.db_path) as conn:
        conn.execute("DELETE FROM query_results")
        conn.commit()
