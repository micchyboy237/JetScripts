"""
02_fallback_strategies.py

Robust fallback strategies and error handling for RAG systems.
Demonstrates graceful degradation when filter extraction or search fails.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FallbackStrategy(str, Enum):
    """Available fallback strategies."""

    DEFAULT_FILTERS = "default_filters"  # Apply sensible defaults
    NO_FILTERS = "no_filters"  # Proceed without filters
    STRICT = "strict"  # Raise error
    BROADEN_SEARCH = "broaden_search"  # Relax filter constraints
    CACHED_RESULTS = "cached_results"  # Return cached results
    DEGRADE_QUALITY = "degrade_quality"  # Lower quality thresholds


class SearchError(Exception):
    """Custom exception for search failures."""

    pass


class FilterExtractionError(SearchError):
    """Error during filter extraction."""

    pass


class InsufficientResultsError(SearchError):
    """Error when too few results found."""

    pass


@dataclass
class Document:
    """Document for demonstration."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RobustRAGRetriever:
    """
    RAG retriever with comprehensive fallback strategies.
    Implements graceful degradation for production reliability.
    """

    def __init__(
        self,
        default_filters: Optional[Dict] = None,
        fallback_strategy: FallbackStrategy = FallbackStrategy.DEFAULT_FILTERS,
        min_results_threshold: int = 2,
        max_fallback_attempts: int = 3,
    ):
        self.default_filters = default_filters or {
            "year": {"$gte": 2020},
            "status": {"$eq": "published"},
        }
        self.fallback_strategy = fallback_strategy
        self.min_results_threshold = min_results_threshold
        self.max_fallback_attempts = max_fallback_attempts
        self.query_cache: Dict[str, List[Dict]] = {}

        # Strategy execution order
        self.strategy_chain = [
            self._try_intelligent_extraction,
            self._try_default_filters,
            self._try_broadened_search,
            self._try_no_filters,
        ]

    def retrieve_with_fallback(
        self,
        query: str,
        k: int = 5,
        strict_mode: bool = False,
    ) -> List[Dict]:
        """
        Retrieve documents with multiple fallback strategies.

        Strategy order:
        1. Intelligent filter extraction
        2. Default filters (if extraction fails or too few results)
        3. Broadened search (relax constraints)
        4. No filters (broadest possible search)
        5. Strict mode: raise error instead of degrading
        """
        attempt = 0

        for strategy in self.strategy_chain:
            if attempt >= self.max_fallback_attempts:
                break

            try:
                logger.info(f"Attempt {attempt + 1}: {strategy.__name__}")
                results = strategy(query, k)

                # Validate results
                if self._validate_results(results):
                    logger.info(
                        f"Success with {strategy.__name__}: {len(results)} results"
                    )
                    return results
                else:
                    logger.warning(
                        f"Too few results ({len(results)}) with {strategy.__name__}"
                    )

            except FilterExtractionError as e:
                logger.error(f"Filter extraction failed: {e}")
                if strict_mode:
                    raise

            except Exception as e:
                logger.error(f"Unexpected error in {strategy.__name__}: {e}")
                if strict_mode:
                    raise SearchError(f"Search failed: {e}")

            attempt += 1

        # All strategies exhausted
        if strict_mode:
            raise InsufficientResultsError(
                "Cannot retrieve relevant documents with available strategies. "
                "Please refine your query."
            )

        # Final fallback: return empty with explanation
        logger.warning("All strategies exhausted, returning empty results")
        return []

    def _try_intelligent_extraction(self, query: str, k: int) -> List[Dict]:
        """Strategy 1: Try LLM-powered filter extraction."""
        try:
            filter_dict = self._extract_filters(query)
            if filter_dict:
                return self._search(query, k, filter_dict)
            else:
                raise FilterExtractionError("No filters extracted")
        except Exception as e:
            raise FilterExtractionError(f"Extraction failed: {e}")

    def _try_default_filters(self, query: str, k: int) -> List[Dict]:
        """Strategy 2: Apply sensible default filters."""
        logger.info("Applying default filters")
        return self._search(query, k, self.default_filters)

    def _try_broadened_search(self, query: str, k: int) -> List[Dict]:
        """Strategy 3: Relax filter constraints."""
        logger.info("Broadening search with relaxed constraints")

        # Example: relax year constraint, remove status filter
        relaxed_filters = {}
        if "year" in self.default_filters:
            relaxed_filters["year"] = {
                "$gte": self.default_filters["year"].get("$gte", 2020) - 2
            }

        return self._search(query, k * 2, relaxed_filters)[:k]

    def _try_no_filters(self, query: str, k: int) -> List[Dict]:
        """Strategy 4: Search without any filters."""
        logger.info("Searching without filters")
        return self._search(query, k * 3, {})[:k]

    def _extract_filters(self, query: str) -> Optional[Dict]:
        """Extract filters from query (simulated)."""
        import re

        query_lower = query.lower()
        conditions = []

        # Simulate occasional extraction failures
        if "error" in query_lower:
            raise FilterExtractionError("Simulated extraction error")

        # Department extraction
        if "technology" in query_lower or "tech" in query_lower:
            conditions.append({"department": "Technology"})

        # Year extraction
        years = re.findall(r"\b(20\d{2})\b", query)
        if years:
            year = int(years[0])
            conditions.append({"year": {"$gte": year, "$lte": year}})

        return {"$and": conditions} if conditions else None

    def _search(
        self, query: str, k: int, filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Simulate vector search."""
        # In production: vectorstore.similarity_search(query, k=k, filter=filter_dict)
        base_results = [
            {"content": f"Result for: {query}", "metadata": {"score": 0.9}},
            {"content": "Another relevant document", "metadata": {"score": 0.8}},
            {"content": "Related information", "metadata": {"score": 0.7}},
        ]
        return base_results[:k]

    def _validate_results(self, results: List[Dict]) -> bool:
        """Validate that results meet minimum quality threshold."""
        return len(results) >= self.min_results_threshold

    def retrieve_with_cache(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve with query caching."""
        import hashlib

        cache_key = hashlib.md5(query.encode()).hexdigest()

        # Check cache
        if cache_key in self.query_cache:
            logger.info("Cache hit!")
            return self.query_cache[cache_key]

        # Perform retrieval
        results = self.retrieve_with_fallback(query, k)

        # Cache results
        self.query_cache[cache_key] = results

        # Limit cache size
        if len(self.query_cache) > 1000:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        return results


def demonstrate_strategy_chain():
    """Demonstrate the fallback strategy chain."""
    print("\n" + "=" * 60)
    print("1. FALLBACK STRATEGY CHAIN")
    print("=" * 60)

    retriever = RobustRAGRetriever()

    test_queries = [
        "What are technology initiatives in 2024?",
        "Show me general information",
        "Simulate error in extraction",
    ]

    for query in test_queries:
        print(f"\n  Query: '{query}'")
        try:
            results = retriever.retrieve_with_fallback(query, k=5)
            print(f"  Results: {len(results)} documents found")
            print(
                f"  Strategy: {'Success' if results else 'Empty (all strategies exhausted)'}"
            )
        except SearchError as e:
            print(f"  Error: {e}")

    print("\n✅ Strategy chain demonstrated")


def demonstrate_strict_mode():
    """Demonstrate strict mode behavior."""
    print("\n" + "=" * 60)
    print("2. STRICT MODE VS GRACEFUL DEGRADATION")
    print("=" * 60)

    # Graceful degradation
    print("\n  Graceful degradation (strict_mode=False):")
    retriever = RobustRAGRetriever()
    results = retriever.retrieve_with_fallback(
        "very specific niche query that might not exist", strict_mode=False
    )
    print(f"    Returns: {len(results)} results (may be empty, no error)")

    # Strict mode
    print("\n  Strict mode (strict_mode=True):")
    try:
        results = retriever.retrieve_with_fallback(
            "simulate error query", strict_mode=True
        )
        print(f"    Returns: {len(results)} results")
    except SearchError as e:
        print(f"    Raises error: {e}")

    print("\n✅ Strict mode vs graceful degradation demonstrated")


def demonstrate_custom_fallback():
    """Demonstrate custom fallback strategy."""
    print("\n" + "=" * 60)
    print("3. CUSTOM FALLBACK STRATEGY")
    print("=" * 60)

    class CustomFallbackRetriever(RobustRAGRetriever):
        """Retriever with custom fallback behavior."""

        def __init__(self):
            super().__init__()
            # Add custom strategy
            self.strategy_chain.insert(1, self._try_synonym_expansion)

        def _try_synonym_expansion(self, query: str, k: int) -> List[Dict]:
            """Try expanding query with synonyms."""
            logger.info("Trying synonym expansion")

            # Simple synonym mapping
            synonyms = {
                "ai": "artificial intelligence",
                "ml": "machine learning",
                "hr": "human resources",
                "tech": "technology",
            }

            expanded_query = query
            for short, full in synonyms.items():
                expanded_query = expanded_query.replace(short, full)

            if expanded_query != query:
                logger.info(f"Expanded query: '{expanded_query}'")
                return self._search(expanded_query, k, self.default_filters)

            raise FilterExtractionError("No synonyms to expand")

    retriever = CustomFallbackRetriever()
    results = retriever.retrieve_with_fallback("What are our AI and ML projects?", k=5)
    print(f"\n  Results: {len(results)} documents found")
    print("  Custom strategy attempted synonym expansion")

    print("\n✅ Custom fallback demonstrated")


def demonstrate_logging_and_monitoring():
    """Demonstrate logging patterns for fallback monitoring."""
    print("\n" + "=" * 60)
    print("4. FALLBACK MONITORING & LOGGING")
    print("=" * 60)

    # Configure detailed logging
    detailed_logger = logging.getLogger("fallback_monitor")
    detailed_logger.setLevel(logging.DEBUG)

    print("\n  Logging levels used:")
    print("    DEBUG - Strategy attempts and decisions")
    print("    INFO  - Successful retrievals")
    print("    WARNING - Strategy fallbacks activated")
    print("    ERROR - Filter extraction failures")

    print("\n  Metrics to monitor:")
    metrics = [
        "fallback_activation_count",
        "strategy_success_rate",
        "average_results_per_strategy",
        "filter_extraction_failure_rate",
        "cache_hit_rate",
        "p99_search_latency",
    ]
    for metric in metrics:
        print(f"    - {metric}")

    print("\n✅ Monitoring patterns documented")


def main():
    """Run all fallback strategy demonstrations."""
    print("=" * 60)
    print("FALLBACK STRATEGIES & ERROR HANDLING")
    print("=" * 60)

    demonstrate_strategy_chain()
    demonstrate_strict_mode()
    demonstrate_custom_fallback()
    demonstrate_logging_and_monitoring()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Robust systems always have fallback strategies.")
    print("   Design for failure, but optimize for success.")


if __name__ == "__main__":
    main()
