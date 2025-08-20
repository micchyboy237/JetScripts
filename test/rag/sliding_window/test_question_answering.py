import pytest
import numpy as np
from question_answering import retrieve_relevant_chunk, answer_question
from utils import sliding_window_split
from sentence_transformers import SentenceTransformer

@pytest.fixture
def sample_text():
    return "AI is advancing. Deep learning excels in image recognition." * 10

@pytest.fixture
def embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

class TestQuestionAnswering:
    def test_retrieve_relevant_chunk(self, sample_text, embedder):
        # Given: A query and sample text
        query = "What does deep learning excel in?"
        chunks = sliding_window_split(sample_text, window_size=100, stride=50)
        expected_chunk = chunks[0]  # First chunk contains relevant info

        # When: Retrieving the relevant chunk
        result = retrieve_relevant_chunk(query, chunks, embedder)

        # Then: Verify the retrieved chunk
        assert result["text"] == expected_chunk["text"]

    def test_answer_question(self, sample_text, mocker):
        # Given: A query and mocked LLM response
        query = "What does deep learning excel in?"
        mocker.patch("mlx_lm.generate", return_value="Deep learning excels in image recognition.")
        expected_answer = "Deep learning excels in image recognition."

        # When: Answering the question
        result = answer_question(sample_text, query)

        # Then: Verify the answer
        assert result == expected_answer