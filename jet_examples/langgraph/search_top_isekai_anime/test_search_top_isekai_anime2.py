# JetScripts/jet_examples/langgraph/test_search_top_isekai_anime.py
import pytest
from langchain.schema import Document
from search_top_isekai_anime2 import GraphState, Anime, AnimeList, generate, grade_generation, web_search
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    return llm


@pytest.fixture
def mock_web_search_tool():
    tool = MagicMock()
    tool.invoke.return_value = [
        {"content": "Anime 1: Description 1\nAnime 2: Description 2"},
        {"content": "Anime 3: Description 3"}
    ]
    return tool


class TestIsekaiAnimeSearch:
    def test_generate_valid_anime_list(self, mock_llm):
        # Given: A valid state with documents and a question
        question = "What are the top 10 isekai anime for 2025?"
        documents = [Document(page_content="Anime 1: Desc 1\nAnime 2: Desc 2")]
        state = GraphState(question=question, documents=documents,
                           generation="", anime_list=None)
        expected_anime_list = AnimeList(animes=[
            Anime(title="Anime 1", description="Desc 1"),
            Anime(title="Anime 2", description="Desc 2"),
            Anime(title="Anime 3", description="Desc 3"),
            Anime(title="Anime 4", description="Desc 4"),
            Anime(title="Anime 5", description="Desc 5"),
            Anime(title="Anime 6", description="Desc 6"),
            Anime(title="Anime 7", description="Desc 7"),
            Anime(title="Anime 8", description="Desc 8"),
            Anime(title="Anime 9", description="Desc 9"),
            Anime(title="Anime 10", description="Desc 10")
        ])
        mock_llm.invoke.return_value = {
            "animes": [
                {"title": f"Anime {i}", "description": f"Desc {i}"} for i in range(1, 11)
            ]
        }

        # When: Generating the anime list
        with patch("langgraph.search_top_isekai_anime.rag_chain", mock_llm):
            result = generate(state)

        # Then: The result should contain a valid anime list with 10 entries
        assert result["anime_list"] == expected_anime_list
        assert len(result["anime_list"].animes) == 10
        assert all(isinstance(anime, Anime)
                   for anime in result["anime_list"].animes)

    def test_generate_incomplete_anime_list(self, mock_llm):
        # Given: A state with an incomplete anime list
        question = "What are the top 10 isekai anime for 2025?"
        documents = [Document(page_content="Anime 1: Desc 1")]
        state = GraphState(question=question, documents=documents,
                           generation="", anime_list=None)
        mock_llm.invoke.return_value = {
            "animes": [{"title": "Anime 1", "description": "Desc 1"}]
        }

        # When: Generating the anime list
        with patch("langgraph.search_top_isekai_anime.rag_chain", mock_llm):
            result = generate(state)

        # Then: The result should indicate an incomplete list
        assert result["anime_list"] is None
        assert result["generation"] == ""

    def test_grade_generation_valid(self, mock_llm):
        # Given: A valid generation with 10 anime
        question = "What are the top 10 isekai anime for 2025?"
        generation = {
            "animes": [{"title": f"Anime {i}", "description": f"Desc {i}"} for i in range(1, 11)]
        }
        state = GraphState(
            question=question, generation=generation, documents=[], anime_list=None)
        expected_score = {"score": "yes"}
        mock_llm.invoke.return_value = expected_score

        # When: Grading the generation
        with patch("langgraph.search_top_isekai_anime.answer_grader", mock_llm):
            result = grade_generation(state)

        # Then: The generation should be graded as useful
        assert result == "useful"
        assert mock_llm.invoke.called

    def test_web_search_success(self, mock_web_search_tool):
        # Given: A valid question for web search
        question = "What are the top 10 isekai anime for 2025?"
        state = GraphState(question=question, documents=[],
                           generation="", anime_list=None)
        expected_docs = [Document(
            page_content="Anime 1: Description 1\nAnime 2: Description 2\nAnime 3: Description 3")]

        # When: Performing web search
        with patch("langgraph.search_top_isekai_anime.web_search_tool", mock_web_search_tool):
            result = web_search(state)

        # Then: The result should contain the expected documents
        assert len(result["documents"]) == 1
        assert result["documents"][0].page_content == expected_docs[0].page_content
        assert result["question"] == question
