from typing import List, Dict, Any
import pytest
from _temp import SpellCorrectedSearchEngine


@pytest.fixture
def search_engine():
    """Fixture to initialize SpellCorrectedSearchEngine."""
    engine = SpellCorrectedSearchEngine()
    documents = [
        {"id": 1, "content": "The quick brown foxx jumps over the lazy dog"},
        {"id": 2, "content": "A beautifull garden blooms with collorful flowers"},
        {"id": 3, "content": "Teh sun sets slowly behind the mountan"},
    ]
    engine.add_documents(documents)
    return engine


def test_search_finds_misspelled_documents(search_engine: SpellCorrectedSearchEngine):
    # Given: A query with correctly spelled keywords
    query = "beautiful garden"
    expected = [
        {"id": 2, "content": "A beautifull garden blooms with collorful flowers"}
    ]

    # When: Performing a search
    results = search_engine.search(query, limit=1)

    # Then: The correct document is returned despite misspellings
    assert len(results) == 1, "Expected exactly one result"
    assert results[0]["id"] == expected[0]["id"], "Expected document ID 2"
    assert results[0]["content"] == expected[0]["content"], "Expected matching content"
    assert results[0]["score"] > 0.5, "Expected high semantic similarity score"


def test_search_handles_multiple_keywords(search_engine: SpellCorrectedSearchEngine):
    # Given: A query for a different keyword
    query = "quick fox"
    expected = [
        {"id": 1, "content": "The quick brown foxx jumps over the lazy dog"}
    ]

    # When: Performing a search
    results = search_engine.search(query, limit=1)

    # Then: The correct document is returned despite misspellings
    assert len(results) == 1, "Expected exactly one result"
    assert results[0]["id"] == expected[0]["id"], "Expected document ID 1"
    assert results[0]["content"] == expected[0]["content"], "Expected matching content"
    assert results[0]["score"] > 0.5, "Expected high semantic similarity score"


def test_spell_correction(search_engine: SpellCorrectedSearchEngine):
    # Given: A text with misspellings
    text = "A beautifull garden"
    expected = "A beautiful garden"

    # When: Correcting the text
    corrected = search_engine.correct_text(text)

    # Then: The text is corrected properly
    assert corrected == expected, f"Expected '{expected}', but got '{corrected}'"
