import pytest
from langgraph.graph import StateGraph
from typing import List
from pydantic import BaseModel, Field
from unittest.mock import MagicMock
from search_top_isekai_anime import GraphState, web_search, generate, anime_chain


class AnimeList(BaseModel):
    titles: List[str] = Field(
        description="List of top 10 isekai anime titles for 2025")


@pytest.fixture
def mock_web_search_tool():
    """Fixture to mock the web search tool."""
    mock = MagicMock()
    mock.invoke.return_value = [
        {"content": "2025 isekai anime: 'Sword Art Online Progressive', 'Re:Zero Season 3', 'Mushoku Tensei Season 3'"},
        {"content": "Top isekai for 2025 includes 'Overlord V', 'That Time I Got Reincarnated as a Slime Season 3'"},
    ]
    return mock


@pytest.fixture
def mock_anime_chain():
    """Fixture to mock the anime chain."""
    mock = MagicMock()
    mock.invoke.return_value = {
        "titles": [
            "Sword Art Online Progressive",
            "Re:Zero Season 3",
            "Mushoku Tensei Season 3",
            "Overlord V",
            "That Time I Got Reincarnated as a Slime Season 3",
            "The Rising of the Shield Hero Season 4",
            "No Game No Life Season 2",
            "KonoSuba Season 4",
            "Log Horizon Season 4",
            "In Another World with My Smartphone Season 3"
        ]
    }
    return mock


class TestIsekaiAnimeSearch:
    def test_web_search_node(self, mock_web_search_tool, monkeypatch):
        """
        Given a query for top isekai anime,
        When the web_search node is executed,
        Then it should return a state with documents containing search results.
        """
        # Arrange
        monkeypatch.setattr(
            "search_top_isekai_anime.web_search_tool", mock_web_search_tool)
        state = GraphState(question="top 10 isekai anime 2025",
                           generation=[], documents=[])
        expected_docs = ["2025 isekai anime: 'Sword Art Online Progressive', 'Re:Zero Season 3', 'Mushoku Tensei Season 3'\nTop isekai for 2025 includes 'Overlord V', 'That Time I Got Reincarnated as a Slime Season 3'"]

        # Act
        result = web_search(state)

        # Assert
        result_docs = [doc.page_content for doc in result["documents"]]
        assert result_docs == expected_docs
        assert result["question"] == state["question"]

    def test_generate_node(self, mock_anime_chain, monkeypatch):
        """
        Given a state with web search documents,
        When the generate node is executed,
        Then it should return a state with a list of 10 anime titles.
        """
        # Arrange
        monkeypatch.setattr(
            "search_top_isekai_anime.anime_chain", mock_anime_chain)
        state = GraphState(
            question="top 10 isekai anime 2025",
            generation=[],
            documents=[{"page_content": "Sample web results"}]
        )
        expected_titles = [
            "Sword Art Online Progressive",
            "Re:Zero Season 3",
            "Mushoku Tensei Season 3",
            "Overlord V",
            "That Time I Got Reincarnated as a Slime Season 3",
            "The Rising of the Shield Hero Season 4",
            "No Game No Life Season 2",
            "KonoSuba Season 4",
            "Log Horizon Season 4",
            "In Another World with My Smartphone Season 3"
        ]

        # Act
        result = generate(state)

        # Assert
        assert result["generation"] == expected_titles
        assert result["question"] == state["question"]
        assert result["documents"] == state["documents"]


@pytest.fixture(scope="module", autouse=True)
def cleanup():
    """Cleanup any resources if needed."""
    yield
    # Add cleanup logic if necessary (e.g., remove temp files)
