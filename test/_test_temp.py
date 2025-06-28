import pytest
from typing import List
from _temp import HTMLNodeSearcher, NodeResult  # Import from the module


class TestHTMLNodeSearcher:
    @pytest.fixture
    def searcher(self) -> HTMLNodeSearcher:
        return HTMLNodeSearcher()

    @pytest.fixture
    def sample_html(self) -> str:
        # Fixed: Removed erroneous parenthesis in HTML string
        return """
        <html>
            <p>Machine learning is a method of data analysis.</p>
            <p>Deep learning uses neural networks with many layers.</p>
            <p>This is a long paragraph about machine learning, deep learning, and AI...</p>
        </html>
        """

    def test_extract_node_texts(self, searcher: HTMLNodeSearcher, sample_html: str):
        # Given: HTML content with multiple paragraphs
        expected = [
            {'text': 'Machine learning is a method of data analysis.', 'tag': 'p'},
            {'text': 'Deep learning uses neural networks with many layers.', 'tag': 'p'},
            {'text': 'This is a long paragraph about machine learning, deep learning, and AI...', 'tag': 'p'}
        ]

        # When: Extracting node texts
        result = searcher.extract_node_texts(sample_html, target_tag='p')

        # Then: Extracted texts and tags match expected
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_chunk_text(self, searcher: HTMLNodeSearcher):
        # Given: A long text to chunk
        long_text = "A" * 600
        expected = ["A" * 512, "A" * 88]

        # When: Chunking the text
        result = searcher.chunk_text(long_text, max_chunk_size=512)

        # Then: Chunks match expected
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_search_nodes(self, searcher: HTMLNodeSearcher, sample_html: str):
        # Given: A query and HTML content
        query = "What is deep learning?"
        expected_texts = [
            'Deep learning uses neural networks with many layers.']

        # When: Searching nodes
        result = searcher.search_nodes(
            query, sample_html, target_tag='p', top_k=1)

        # Then: Top result matches expected text and has a valid similarity score
        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
        assert result[0]['text'] == expected_texts[
            0], f"Expected text {expected_texts[0]}, got {result[0]['text']}"
        assert 0 <= result[0][
            'similarity'] <= 1, f"Similarity score {result[0]['similarity']} out of range"
