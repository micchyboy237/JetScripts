import pytest
from utils import sliding_window_split, TextChunk

@pytest.fixture
def sample_text():
    return "This is a test document about AI. AI is transforming industries. It has many applications." * 10

class TestSlidingWindowSplit:
    def test_sliding_window_split(self, sample_text):
        # Given: A sample text and window parameters
        window_size = 100
        stride = 50
        expected_chunks = [
            {"text": sample_text[:100], "start_idx": 0, "end_idx": 100},
            {"text": sample_text[50:150], "start_idx": 50, "end_idx": 150},
        ]
        expected_chunk_count = (len(sample_text) - window_size) // stride + (1 if len(sample_text) % stride else 0)

        # When: Splitting the text
        result = sliding_window_split(sample_text, window_size, stride)

        # Then: Verify the number of chunks and their content
        assert len(result) == expected_chunk_count
        for i, chunk in enumerate(result[:2]):  # Test first two chunks
            assert chunk == expected_chunks[i]