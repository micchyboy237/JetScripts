from urllib.parse import urlparse
import pytest
from typing import List, Dict
from unittest.mock import patch
from jet.llm.utils.bm25_plus import bm25_plus
from run_get_diverse_links import preprocess_urls


class TestRerankUrls:
    @patch('rerank_urls.clean_url')
    @patch('rerank_urls.parse_url')
    @patch('rerank_urls.load_file')
    @patch('rerank_urls.save_file')
    def test_url_mapping(self, mock_save_file, mock_load_file, mock_parse_url, mock_clean_url):
        # Mock dependencies
        mock_load_file.return_value = [
            "https://www.ranker.com/list/new-and-upcoming-fantasy-shows-2024/molly-gander",
            "https://www.ranker.com/list/new-and-upcoming-fantasy-shows-2024/molly-gander",  # Duplicate
            "https://peacedoorball.blog/en/upcoming-isekai-anime-releases-for-2025-all-announced-titles-so-far",
            "https://www.ranker.com/list/upcoming-anime-2025/anna-lindwasser",
            "https://example.com/image.jpg"  # Filtered out
        ]
        mock_clean_url.side_effect = lambda x: x
        mock_parse_url.side_effect = lambda x: [urlparse(x).scheme, urlparse(
            x).netloc.replace('.', ' '), *urlparse(x).path.lstrip('/').split('/')]

        # Input data
        urls = mock_load_file.return_value
        query = "isekai 2025"
        top_k = 3

        # Expected outputs
        expected_preprocessed = [
            "https www ranker com list new and upcoming fantasy shows 2024 molly gander",
            "https peacedoorball blog en upcoming isekai anime releases for 2025 all announced titles so far",
            "https www ranker com list upcoming anime 2025 anna lindwasser"
        ]
        expected_mapping = {
            0: "https://www.ranker.com/list/new-and-upcoming-fantasy-shows-2024/molly-gander",
            1: "https://peacedoorball.blog/en/upcoming-isekai-anime-releases-for-2025-all-announced-titles-so-far",
            2: "https://www.ranker.com/list/upcoming-anime-2025/anna-lindwasser"
        }
        expected_urls = list(expected_mapping.values())

        # Run preprocessing and BM25+
        result_preprocessed, result_mapping = preprocess_urls(urls)
        result_bm25 = bm25_plus(result_preprocessed, query)
        result_urls = [result_mapping[result["doc_index"]]
                       for result in result_bm25 if result["doc_index"] in result_mapping]
        result_urls = list(set(result_urls))[:top_k]

        # Assertions
        assert result_preprocessed == expected_preprocessed, f"Expected preprocessed URLs {expected_preprocessed}, got {result_preprocessed}"
        assert result_mapping == expected_mapping, f"Expected mapping {expected_mapping}, got {result_mapping}"
        assert sorted(result_urls) == sorted(
            expected_urls), f"Expected URLs {expected_urls}, got {result_urls}"

    @patch('rerank_urls.clean_url')
    @patch('rerank_urls.parse_url')
    def test_empty_urls(self, mock_parse_url, mock_clean_url):
        # Test empty URL list
        urls: List[str] = []
        query = "isekai 2025"
        expected_preprocessed: List[str] = []
        expected_mapping: Dict[int, str] = {}
        expected_urls: List[str] = []

        result_preprocessed, result_mapping = preprocess_urls(urls)
        result_bm25 = bm25_plus(result_preprocessed, query)
        result_urls = [result_mapping[result["doc_index"]]
                       for result in result_bm25 if result["doc_index"] in result_mapping]
        result_urls = list(set(result_urls))

        assert result_preprocessed == expected_preprocessed, f"Expected preprocessed URLs {expected_preprocessed}, got {result_preprocessed}"
        assert result_mapping == expected_mapping, f"Expected mapping {expected_mapping}, got {result_mapping}"
        assert result_urls == expected_urls, f"Expected URLs {expected_urls}, got {result_urls}"

    @patch('rerank_urls.clean_url')
    @patch('rerank_urls.parse_url')
    def test_all_filtered_urls(self, mock_parse_url, mock_clean_url):
        # Test URLs that are all filtered out
        urls = ["https://example.com/image.jpg", "https://example.com/wp-json"]
        mock_clean_url.side_effect = lambda x: x
        mock_parse_url.side_effect = lambda x: [urlparse(x).scheme, urlparse(
            x).netloc.replace('.', ' '), *urlparse(x).path.lstrip('/').split('/')]
        query = "isekai 2025"
        expected_preprocessed: List[str] = []
        expected_mapping: Dict[int, str] = {}
        expected_urls: List[str] = []

        result_preprocessed, result_mapping = preprocess_urls(urls)
        result_bm25 = bm25_plus(result_preprocessed, query)
        result_urls = [result_mapping[result["doc_index"]]
                       for result in result_bm25 if result["doc_index"] in result_mapping]
        result_urls = list(set(result_urls))

        assert result_preprocessed == expected_preprocessed, f"Expected preprocessed URLs {expected_preprocessed}, got {result_preprocessed}"
        assert result_mapping == expected_mapping, f"Expected mapping {expected_mapping}, got {result_mapping}"
        assert result_urls == expected_urls, f"Expected URLs {expected_urls}, got {result_urls}"
