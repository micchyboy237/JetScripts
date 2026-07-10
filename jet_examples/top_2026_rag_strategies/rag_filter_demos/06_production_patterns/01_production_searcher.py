"""
01_production_searcher.py

Complete production-ready RAG searcher with layered filtering.
Combines security, intelligent extraction, credibility, and vector search.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilterLayer(Enum):
    """Filter layers in order of application."""

    SECURITY = 1  # Applied first - non-negotiable
    COMPLIANCE = 2  # Regulatory requirements
    INTELLIGENT = 3  # LLM-extracted filters
    CREDIBILITY = 4  # Quality thresholds
    VECTOR = 5  # Similarity search (always last)


@dataclass
class SearchConfig:
    """Configuration for the production searcher."""

    default_k: int = 5
    min_credibility: float = 0.7
    use_intelligent_filtering: bool = True
    use_caching: bool = True
    cache_ttl_seconds: int = 3600
    fallback_strategy: str = "default_filters"
    max_filter_conditions: int = 10
    enable_audit_logging: bool = True


@dataclass
class SearchResult:
    """Structured search result with metadata."""

    documents: List[Dict]
    applied_filters: Dict[str, Any]
    filter_layers_applied: List[FilterLayer]
    total_candidates: int
    final_count: int
    search_time_ms: float
    from_cache: bool = False


@dataclass
class Document:
    """Document with metadata for demonstration."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentRAGSearcher:
    """
    Production-ready RAG searcher with comprehensive filtering.

    Filter precedence (in order):
    1. User access controls (security) - NEVER skip
    2. Compliance requirements - NEVER skip
    3. Intelligent metadata extraction (if enabled)
    4. Credibility thresholds
    5. Vector similarity search
    """

    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self._search_cache: Dict[str, Tuple[float, List[Dict]]] = {}

        # Metadata field definitions
        self.metadata_fields = {
            "department": "Document department (string)",
            "year": "Publication year (integer)",
            "doc_type": "Document type (string)",
            "credibility_score": "Source credibility 0-1 (float)",
            "topic": "Document topic (string)",
            "author": "Document author (string)",
            "status": "Document status (string)",
            "classification": "Security classification (string)",
        }

        logger.info(f"Initialized IntelligentRAGSearcher with config: {self.config}")

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        min_credibility: Optional[float] = None,
        k: Optional[int] = None,
        use_intelligent_filtering: Optional[bool] = None,
        additional_filters: Optional[Dict] = None,
    ) -> SearchResult:
        """
        Comprehensive search with multiple filter layers.

        Args:
            query: Natural language search query
            user_id: User identifier for security filtering
            min_credibility: Override default credibility threshold
            k: Number of results to return
            use_intelligent_filtering: Override intelligent filtering setting
            additional_filters: Additional filters to apply

        Returns:
            SearchResult with documents and filtering metadata
        """
        import time

        start_time = time.time()

        k = k or self.config.default_k
        min_cred = min_credibility or self.config.min_credibility
        use_intelligent = (
            use_intelligent_filtering
            if use_intelligent_filtering is not None
            else self.config.use_intelligent_filtering
        )

        # Check cache
        if self.config.use_caching:
            cache_key = self._build_cache_key(query, user_id, k)
            cached = self._check_cache(cache_key)
            if cached:
                elapsed = (time.time() - start_time) * 1000
                return SearchResult(
                    documents=cached,
                    applied_filters={},
                    filter_layers_applied=[FilterLayer.VECTOR],
                    total_candidates=len(cached),
                    final_count=len(cached),
                    search_time_ms=elapsed,
                    from_cache=True,
                )

        # Build filter layers
        applied_layers = []
        all_filters = []

        # Layer 1: Security (always applied if user_id provided)
        if user_id:
            security_filter = self._get_security_filter(user_id)
            if security_filter:
                all_filters.append(security_filter)
                applied_layers.append(FilterLayer.SECURITY)
                logger.debug(f"Applied security filter for user {user_id}")

        # Layer 2: Compliance (would check user's jurisdiction)
        compliance_filter = self._get_compliance_filter(user_id)
        if compliance_filter:
            all_filters.append(compliance_filter)
            applied_layers.append(FilterLayer.COMPLIANCE)

        # Layer 3: Intelligent filtering
        if use_intelligent:
            try:
                intelligent_filter = self._extract_intelligent_filters(query)
                if intelligent_filter:
                    all_filters.append(intelligent_filter)
                    applied_layers.append(FilterLayer.INTELLIGENT)
                    logger.debug(f"Applied intelligent filters: {intelligent_filter}")
            except Exception as e:
                logger.warning(f"Intelligent filter extraction failed: {e}")

        # Layer 4: Credibility
        credibility_filter = {"credibility_score": {"$gte": min_cred}}
        all_filters.append(credibility_filter)
        applied_layers.append(FilterLayer.CREDIBILITY)

        # Layer 5: Additional filters
        if additional_filters:
            all_filters.append(additional_filters)

        # Merge all filters
        merged_filter = self._merge_filters(all_filters)

        # Simulate search (in production, this calls vectorstore.similarity_search)
        documents = self._simulate_search(query, merged_filter, k)

        # Cache results
        if self.config.use_caching:
            cache_key = self._build_cache_key(query, user_id, k)
            self._cache_results(cache_key, documents)

        elapsed = (time.time() - start_time) * 1000

        result = SearchResult(
            documents=documents,
            applied_filters={
                "merged_filter": merged_filter,
                "credibility_threshold": min_cred,
                "user_id": user_id,
            },
            filter_layers_applied=applied_layers,
            total_candidates=len(documents) * 2,  # Simulated
            final_count=len(documents),
            search_time_ms=elapsed,
        )

        if self.config.enable_audit_logging:
            self._audit_log(query, user_id, result)

        return result

    def _get_security_filter(self, user_id: str) -> Dict:
        """Get user-specific security filters."""
        # In production: fetch from auth/IAM system
        permissions = self._get_user_permissions(user_id)

        if not permissions:
            return {"visibility": {"$eq": "public"}}

        conditions = [
            {"department": {"$in": permissions.get("allowed_departments", [])}},
            {
                "clearance_level": {
                    "$in": permissions.get("clearance_levels", ["public"])
                }
            },
        ]

        if permissions.get("excluded_projects"):
            for proj in permissions["excluded_projects"]:
                conditions.append({"project": {"$ne": proj}})

        return {"$and": conditions}

    def _get_compliance_filter(self, user_id: str) -> Optional[Dict]:
        """Get compliance-related filters."""
        # In production: check user's jurisdiction and apply relevant regulations
        return None  # Simplified for demonstration

    def _extract_intelligent_filters(self, query: str) -> Optional[Dict]:
        """Extract filters from query using LLM."""
        # In production: call LLM for filter extraction
        query_lower = query.lower()
        conditions = []

        # Simple pattern matching for demonstration
        department_keywords = {
            "technology": "Technology",
            "tech": "Technology",
            "hr": "HR",
            "finance": "Finance",
            "legal": "Legal",
            "marketing": "Marketing",
        }

        for keyword, dept in department_keywords.items():
            if keyword in query_lower:
                conditions.append({"department": dept})
                break

        # Year extraction
        import re

        years = re.findall(r"\b(20\d{2})\b", query)
        if len(years) == 1:
            conditions.append({"year": {"$eq": int(years[0])}})
        elif len(years) >= 2:
            min_year = min(int(y) for y in years)
            max_year = max(int(y) for y in years)
            conditions.append({"year": {"$gte": min_year, "$lte": max_year}})

        if not conditions:
            return None

        return {"$and": conditions} if len(conditions) > 1 else conditions[0]

    @lru_cache(maxsize=1000)
    def _get_user_permissions(self, user_id: str) -> Dict:
        """Get cached user permissions."""
        # In production: call auth service
        return {
            "allowed_departments": ["Technology", "Engineering", "Product"],
            "clearance_levels": ["public", "internal", "confidential"],
            "excluded_projects": [],
        }

    def _merge_filters(self, filters: List[Dict]) -> Dict:
        """Merge multiple filter dicts with AND logic."""
        valid_filters = [f for f in filters if f]
        if not valid_filters:
            return {}
        elif len(valid_filters) == 1:
            return valid_filters[0]
        else:
            return {"$and": valid_filters}

    def _simulate_search(self, query: str, filter_dict: Dict, k: int) -> List[Dict]:
        """Simulate vector search (replace with actual vectorstore call)."""
        # In production: vectorstore.similarity_search(query, k=k, filter=filter_dict)
        sample_docs = [
            {
                "content": f"Result for '{query}': AI initiatives in 2024",
                "metadata": {"department": "Technology", "year": 2024, "score": 0.95},
            },
            {
                "content": f"Result for '{query}': Cloud migration strategy",
                "metadata": {"department": "Technology", "year": 2024, "score": 0.88},
            },
            {
                "content": f"Result for '{query}': Machine learning deployment",
                "metadata": {"department": "Engineering", "year": 2023, "score": 0.82},
            },
        ]
        return sample_docs[:k]

    def _build_cache_key(self, query: str, user_id: str, k: int) -> str:
        """Build cache key from query parameters."""
        key_data = f"{query}:{user_id}:{k}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Check if results are cached and still valid."""
        if cache_key in self._search_cache:
            timestamp, results = self._search_cache[cache_key]
            age = datetime.now().timestamp() - timestamp
            if age < self.config.cache_ttl_seconds:
                logger.info(f"Cache hit for key: {cache_key[:8]}...")
                return results
            else:
                del self._search_cache[cache_key]
        return None

    def _cache_results(self, cache_key: str, results: List[Dict]):
        """Cache search results."""
        self._search_cache[cache_key] = (datetime.now().timestamp(), results)

        # Cleanup old cache entries
        if len(self._search_cache) > 1000:
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]

    def _audit_log(self, query: str, user_id: str, result: SearchResult):
        """Log search for audit purposes."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "result_count": result.final_count,
            "search_time_ms": result.search_time_ms,
            "filters_applied": [layer.name for layer in result.filter_layers_applied],
            "from_cache": result.from_cache,
        }
        logger.info(f"AUDIT: {json.dumps(audit_entry)}")


def demonstrate_layered_search():
    """Demonstrate the layered search approach."""
    print("\n" + "=" * 60)
    print("1. LAYERED SEARCH DEMONSTRATION")
    print("=" * 60)

    searcher = IntelligentRAGSearcher()

    queries = [
        ("What are our AI plans for 2024?", "user_001"),
        ("Show me technology reports", None),  # Anonymous user
        ("Finance policies from 2023", "user_002"),
    ]

    for query, user_id in queries:
        result = searcher.search(query=query, user_id=user_id)

        print(f"\n  Query: '{query}'")
        print(f"  User: {user_id or 'Anonymous'}")
        print(f"  Results: {result.final_count} documents")
        print(f"  Layers applied: {[l.name for l in result.filter_layers_applied]}")
        print(f"  Time: {result.search_time_ms:.1f}ms")
        print(f"  Cached: {result.from_cache}")

        if result.documents:
            print(f"  Top result: {result.documents[0]['content'][:60]}...")

    print("\n✅ Layered search complete")


def demonstrate_filter_precedence():
    """Demonstrate filter layer precedence."""
    print("\n" + "=" * 60)
    print("2. FILTER PRECEDENCE VISUALIZATION")
    print("=" * 60)

    print("\n  Filter application order:")
    for layer in FilterLayer:
        print(f"    {layer.value}. {layer.name}:")
        if layer == FilterLayer.SECURITY:
            print("       └─ User permissions (MANDATORY)")
        elif layer == FilterLayer.COMPLIANCE:
            print("       └─ Regulatory requirements (MANDATORY)")
        elif layer == FilterLayer.INTELLIGENT:
            print("       └─ LLM-extracted filters (OPTIONAL)")
        elif layer == FilterLayer.CREDIBILITY:
            print("       └─ Quality thresholds (CONFIGURABLE)")
        elif layer == FilterLayer.VECTOR:
            print("       └─ Similarity search (ALWAYS LAST)")

    print("\n✅ Filter precedence explained")


def demonstrate_configuration():
    """Demonstrate different configurations."""
    print("\n" + "=" * 60)
    print("3. CONFIGURATION OPTIONS")
    print("=" * 60)

    configs = [
        ("Default", SearchConfig()),
        (
            "High Security",
            SearchConfig(
                min_credibility=0.9,
                use_intelligent_filtering=False,
                use_caching=False,
                enable_audit_logging=True,
            ),
        ),
        (
            "Performance",
            SearchConfig(
                use_caching=True,
                cache_ttl_seconds=7200,
                use_intelligent_filtering=False,
                min_credibility=0.5,
            ),
        ),
        (
            "Maximum Intelligence",
            SearchConfig(
                use_intelligent_filtering=True,
                min_credibility=0.8,
                max_filter_conditions=20,
            ),
        ),
    ]

    for name, config in configs:
        print(f"\n  {name}:")
        print(f"    k: {config.default_k}")
        print(f"    Min credibility: {config.min_credibility}")
        print(f"    Intelligent filtering: {config.use_intelligent_filtering}")
        print(f"    Caching: {config.use_caching}")
        print(f"    Audit logging: {config.enable_audit_logging}")

    print("\n✅ Configuration options demonstrated")


def main():
    """Run all production searcher demonstrations."""
    print("=" * 60)
    print("PRODUCTION RAG SEARCHER DEMONSTRATION")
    print("=" * 60)

    demonstrate_layered_search()
    demonstrate_filter_precedence()
    demonstrate_configuration()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Production searchers must layer filters in the")
    print("   correct order: Security → Compliance → Intelligence → Quality → Vector")


if __name__ == "__main__":
    main()
