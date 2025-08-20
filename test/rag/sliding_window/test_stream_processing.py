import pytest
from stream_processing import StreamProcessor

@pytest.fixture
def processor():
    return StreamProcessor(window_size=100, stride=50)

class TestStreamProcessor:
    def test_process_stream(self, processor, mocker):
        # Given: A stream of text and mocked LLM response
        new_text = "AI is transforming industries. " * 10
        expected_summary = "AI is transforming industries."
        mocker.patch("mlx_lm.generate", return_value=expected_summary)

        # When: Processing the stream
        result = processor.process_stream(new_text)

        # Then: Verify the summary
        assert result == expected_summary

    def test_process_stream_insufficient_text(self, processor):
        # Given: A short text that doesn't meet window size
        new_text = "Short text."

        # When: Processing the stream
        result = processor.process_stream(new_text)

        # Then: Verify no summary is generated
        assert result is None