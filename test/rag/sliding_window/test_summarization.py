import pytest
from summarization import summarize_document

@pytest.fixture
def sample_text():
    return "This is a test document about AI. AI is transforming industries. It has many applications." * 10

class TestSummarization:
    def test_summarize_document(self, sample_text, mocker):
        # Given: A mocked LLM response
        mocker.patch("mlx_lm.generate", side_effect=["Chunk summary 1.", "Chunk summary 2.", "Final summary."])
        expected_summary = "Final summary."

        # When: Summarizing the document
        result = summarize_document(sample_text)

        # Then: Verify the final summary
        assert result == expected_summary