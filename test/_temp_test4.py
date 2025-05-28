from typing import List, Dict
import re
from difflib import SequenceMatcher
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import pytest

# Download required NLTK data (run once on Mac M1)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')


class LinkSearcher:
    def __init__(self, links: List[Dict[str, str]]):
        """
        Initialize with a list of links, each link being a dict with 'url' and 'title'
        Example link: {'url': 'http://example.com', 'title': 'Example Site'}
        """
        self.links = links
        self.keyword_weights = {
            "tiktok": 2.0,
            "live selling": 1.5,
            "philippines": 1.5,
            "registration": 1.2,
            "guidelines": 1.2,
            "2025": 2.0  # Weight for year, if present in query
        }

    def _parse_query(self, query: str) -> List[str]:
        """Parse query into sub-queries based on 'or' operator."""
        query = query.lower().strip()
        sub_queries = [q.strip() for q in query.split(" or ")]
        return [q for q in sub_queries if q]

    def _extract_year(self, query: str) -> str | None:
        """Extract a four-digit year from the query, if present."""
        match = re.search(r'\b\d{4}\b', query)
        return match.group(0) if match else None

    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower().replace('_', ' '))
        return list(synonyms)

    def _tokenize_and_expand(self, query: str) -> List[str]:
        """Tokenize query and expand with synonyms for key terms."""
        tokens = word_tokenize(query)
        expanded = []
        for token in tokens:
            expanded.append(token)
            if token in self.keyword_weights:
                expanded.extend(self._get_synonyms(token))
        return expanded

    def search(self, query: str, method: str = "combined") -> List[Dict[str, str]]:
        """
        Search links based on query using specified method.
        Methods: 'exact', 'partial', 'fuzzy', 'combined'
        """
        if not query.strip():
            return []

        sub_queries = self._parse_query(query)
        year = self._extract_year(query)
        results = []

        for sub_query in sub_queries:
            if method == "exact":
                results.extend(self._exact_match(sub_query, year))
            elif method == "partial":
                results.extend(self._partial_match(sub_query, year))
            elif method == "fuzzy":
                results.extend(self._fuzzy_match(
                    sub_query, year, threshold=0.6))
            else:  # combined
                results.extend(self._combined_search(sub_query, year))

        # Remove duplicates and sort by relevance
        seen_urls = set()
        unique_results = []
        for link in results:
            if link['url'] not in seen_urls:
                unique_results.append(link)
                seen_urls.add(link['url'])

        return self._rank_results(unique_results, sub_queries)

    def _exact_match(self, query: str, year: str | None) -> List[Dict[str, str]]:
        """Exact string matching in title with optional year filter."""
        if year:
            return [
                link for link in self.links
                if query in link['title'].lower() and year in link['title'].lower()
            ]
        return [
            link for link in self.links
            if query in link['title'].lower()
        ]

    def _partial_match(self, query: str, year: str | None) -> List[Dict[str, str]]:
        """Partial matching using regex with optional year filter."""
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        if year:
            return [
                link for link in self.links
                if pattern.search(link['title']) and year in link['title'].lower()
            ]
        return [
            link for link in self.links
            if pattern.search(link['title'])
        ]

    def _fuzzy_match(self, query: str, year: str | None, threshold: float) -> List[Dict[str, str]]:
        """Fuzzy matching using SequenceMatcher with optional year filter."""
        results = []
        expanded_query = self._tokenize_and_expand(query)
        for link in self.links:
            if year and year not in link['title'].lower():
                continue
            max_ratio = max(
                SequenceMatcher(None, q, link['title'].lower()).ratio()
                for q in expanded_query
            )
            if max_ratio >= threshold:
                results.append((link, max_ratio))
        return [link for link, _ in sorted(results, key=lambda x: x[1], reverse=True)]

    def _combined_search(self, query: str, year: str | None) -> List[Dict[str, str]]:
        """Combined approach: exact matches first, then fuzzy matches with optional year filter."""
        exact_results = self._exact_match(query, year)
        fuzzy_results = self._fuzzy_match(query, year, threshold=0.6)

        seen_urls = set()
        combined = []

        for link in exact_results:
            if link['url'] not in seen_urls:
                combined.append(link)
                seen_urls.add(link['url'])

        for link in fuzzy_results:
            if link['url'] not in seen_urls:
                combined.append(link)
                seen_urls.add(link['url'])

        return combined

    def _rank_results(self, results: List[Dict[str, str]], queries: List[str]) -> List[Dict[str, str]]:
        """Rank results based on keyword coverage and weights."""
        scored_results = []
        for link in results:
            score = 0.0
            title = link['title'].lower()
            for query in queries:
                tokens = word_tokenize(query)
                for token in tokens:
                    if token in title:
                        score += self.keyword_weights.get(token, 1.0)
            scored_results.append((link, score))
        return [link for link, _ in sorted(scored_results, key=lambda x: x[1], reverse=True)]


class TestLinkSearcher:
    links = [
        {'url': 'http://tiktok.com/ph1',
            'title': '2025 TikTok Live Selling Registration Philippines'},
        {'url': 'http://tiktok.com/ph2', 'title': 'TikTok Seller Guidelines 2025'},
        {'url': 'http://example.com', 'title': 'General E-commerce Tips'},
        {'url': 'http://tiktok.com/ph3',
            'title': '2024 TikTok Live Selling Guide Philippines'},
        {'url': 'http://coding.com', 'title': 'Learn Coding Online'}
    ]

    def test_empty_query(self):
        """Test searching with empty query."""
        searcher = LinkSearcher(self.links)
        result = searcher.search("")
        expected = []
        assert result == expected, f"Expected {expected}, got {result}"

    def test_exact_match_with_year(self):
        """Test exact matching with year in query."""
        searcher = LinkSearcher(self.links)
        result = searcher.search("tiktok live selling 2025", method="exact")
        expected = [
            {'url': 'http://tiktok.com/ph1',
                'title': '2025 TikTok Live Selling Registration Philippines'},
            {'url': 'http://tiktok.com/ph2', 'title': 'TikTok Seller Guidelines 2025'}
        ]
        assert result == expected, f"Expected {expected}, got {result}"

    def test_exact_match_no_year(self):
        """Test exact matching without year in query."""
        searcher = LinkSearcher(self.links)
        result = searcher.search("tiktok live selling", method="exact")
        expected = [
            {'url': 'http://tiktok.com/ph1',
                'title': '2025 TikTok Live Selling Registration Philippines'},
            {'url': 'http://tiktok.com/ph2',
                'title': 'TikTok Seller Guidelines 2025'},
            {'url': 'http://tiktok.com/ph3',
                'title': '2024 TikTok Live Selling Guide Philippines'}
        ]
        assert result == expected, f"Expected {expected}, got {result}"

    def test_partial_match_with_year(self):
        """Test partial matching with year in query."""
        searcher = LinkSearcher(self.links)
        result = searcher.search("registration 2025", method="partial")
        expected = [
            {'url': 'http://tiktok.com/ph1',
                'title': '2025 TikTok Live Selling Registration Philippines'}
        ]
        assert result == expected, f"Expected {expected}, got {result}"

    def test_fuzzy_match_with_typo(self):
        """Test fuzzy matching with typo and year in query."""
        searcher = LinkSearcher(self.links)
        result = searcher.search("tik tok sell 2025", method="fuzzy")
        expected = [
            {'url': 'http://tiktok.com/ph1',
                'title': '2025 TikTok Live Selling Registration Philippines'},
            {'url': 'http://tiktok.com/ph2', 'title': 'TikTok Seller Guidelines 2025'}
        ]
        assert result == expected, f"Expected {expected}, got {result}"

    def test_multi_part_query(self):
        """Test handling of multi-part query with 'or'."""
        searcher = LinkSearcher(self.links)
        query = "tiktok live selling or seller guidelines 2025"
        result = searcher.search(query)
        expected = [
            {'url': 'http://tiktok.com/ph1',
                'title': '2025 TikTok Live Selling Registration Philippines'},
            {'url': 'http://tiktok.com/ph2', 'title': 'TikTok Seller Guidelines 2025'}
        ]
        assert result == expected, f"Expected {expected}, got {result}"

    def test_synonym_handling(self):
        """Test synonym expansion in fuzzy matching."""
        searcher = LinkSearcher(self.links)
        result = searcher.search("signup 2025", method="fuzzy")
        expected = [
            {'url': 'http://tiktok.com/ph1',
                'title': '2025 TikTok Live Selling Registration Philippines'}
        ]
        assert result == expected, f"Expected {expected}, got {result}"

    def test_no_year_query(self):
        """Test query without year reference."""
        searcher = LinkSearcher(self.links)
        result = searcher.search("tiktok guidelines")
        expected = [
            {'url': 'http://tiktok.com/ph2', 'title': 'TikTok Seller Guidelines 2025'}
        ]
        assert result == expected, f"Expected {expected}, got {result}"


if __name__ == "__main__":
    import os
    from jet.file.utils import load_file, save_file

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"

    query = "Tips and links to 2025 online registration steps for TikTok live selling in the Philippines, or recent guidelines for online sellers on TikTok in the Philippines, or 2025 registration process for TikTok live selling in the Philippines."

    # Load JSON data
    docs = load_file(docs_file)
    print(f"Loaded JSON data {len(docs)} from: {docs_file}")
    links = [{"url": doc_link["url"], "title": doc_link["text"]}
             for doc in docs for doc_link in doc["metadata"]["links"]]
    save_file(links, f"{output_dir}/links.json")

    searcher = LinkSearcher(links)
    results = searcher.search(query)

    save_file({
        "query": query,
        "results": results
    }, f"{output_dir}/results.json")
