"""
03_elasticsearch_filtering.py

Elasticsearch vector search with metadata filtering.
Demonstrates combining traditional search with vector similarity.
"""

from datetime import datetime
from typing import Any, Dict, List


class ElasticsearchFilterBuilder:
    """
    Build Elasticsearch queries with vector search and metadata filters.
    Uses the bool query structure with must/should/filter/must_not clauses.
    """

    @staticmethod
    def term_filter(field: str, value: Any) -> Dict:
        """Exact term match filter."""
        return {"term": {field: value}}

    @staticmethod
    def terms_filter(field: str, values: List[Any]) -> Dict:
        """Match any of the terms."""
        return {"terms": {field: values}}

    @staticmethod
    def range_filter(
        field: str, gte: Any = None, gt: Any = None, lte: Any = None, lt: Any = None
    ) -> Dict:
        """Range filter."""
        range_dict = {}
        if gte is not None:
            range_dict["gte"] = gte
        if gt is not None:
            range_dict["gt"] = gt
        if lte is not None:
            range_dict["lte"] = lte
        if lt is not None:
            range_dict["lt"] = lt
        return {"range": {field: range_dict}}

    @staticmethod
    def knn_query(
        field: str, query_vector: List[float], k: int = 10, num_candidates: int = 100
    ) -> Dict:
        """K-nearest neighbors vector search."""
        return {
            "knn": {
                "field": field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
            }
        }

    @staticmethod
    def bool_query(
        must: List[Dict] = None,
        should: List[Dict] = None,
        filter_clauses: List[Dict] = None,
        must_not: List[Dict] = None,
        minimum_should_match: int = None,
    ) -> Dict:
        """Boolean query combining multiple clauses."""
        bool_dict = {}
        if must:
            bool_dict["must"] = must
        if should:
            bool_dict["should"] = should
            if minimum_should_match:
                bool_dict["minimum_should_match"] = minimum_should_match
        if filter_clauses:
            bool_dict["filter"] = filter_clauses
        if must_not:
            bool_dict["must_not"] = must_not
        return {"bool": bool_dict}

    @staticmethod
    def build_search_query(
        query_vector: List[float],
        filters: List[Dict] = None,
        must_not: List[Dict] = None,
        size: int = 10,
        source_fields: List[str] = None,
    ) -> Dict:
        """Build complete Elasticsearch search query."""
        query = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        ElasticsearchFilterBuilder.knn_query("embedding", query_vector)
                    ]
                }
            },
        }

        if filters:
            query["query"]["bool"]["filter"] = filters

        if must_not:
            query["query"]["bool"]["must_not"] = must_not

        if source_fields:
            query["_source"] = source_fields

        return query


def demonstrate_verified_sources():
    """Demonstrate filtering by verified sources."""
    print("\n" + "=" * 60)
    print("1. VERIFIED ACADEMIC SOURCES")
    print("=" * 60)

    fb = ElasticsearchFilterBuilder

    # Simulated query vector
    query_vector = [0.1] * 384

    query = fb.build_search_query(
        query_vector=query_vector,
        filters=[
            fb.term_filter("source_verified", True),
            fb.terms_filter("source_type", ["peer_reviewed", "official", "academic"]),
            fb.range_filter("citation_count", gte=10),
        ],
        size=10,
    )

    print("\n  Query structure:")
    print("  - Vector search on 'embedding' field")
    print("  - Only verified sources")
    print("  - Source types: peer_reviewed, official, academic")
    print("  - Minimum 10 citations")
    print("\n  Full query:")
    import json

    print(f"  {json.dumps(query, indent=4)}")
    print("\n✅ Verified source filter")


def demonstrate_date_range_filtering():
    """Demonstrate date range filtering with vector search."""
    print("\n" + "=" * 60)
    print("2. RECENT PUBLICATIONS: Last 2 years")
    print("=" * 60)

    fb = ElasticsearchFilterBuilder
    query_vector = [0.1] * 384
    current_year = datetime.now().year

    query = fb.build_search_query(
        query_vector=query_vector,
        filters=[
            fb.range_filter("publication_date", gte=f"{current_year - 2}-01-01"),
            fb.term_filter("status", "published"),
            fb.terms_filter("department", ["Technology", "Research"]),
        ],
    )

    print("\n  Query structure:")
    print(f"  - Date range: {current_year - 2}-01-01 to present")
    print("  - Published documents only")
    print("  - Technology and Research departments")
    print("\n✅ Date range filter")


def demonstrate_exclusion_patterns():
    """Demonstrate exclusion patterns (must_not)."""
    print("\n" + "=" * 60)
    print("3. EXCLUDE DRAFTS AND RESTRICTED CONTENT")
    print("=" * 60)

    fb = ElasticsearchFilterBuilder
    query_vector = [0.1] * 384

    query = fb.build_search_query(
        query_vector=query_vector,
        filters=[fb.range_filter("credibility_score", gte=0.7)],
        must_not=[
            fb.term_filter("status", "draft"),
            fb.term_filter("classification", "restricted"),
            fb.term_filter("classification", "secret"),
        ],
    )

    print("\n  Query structure:")
    print("  - Minimum credibility score: 0.7")
    print("  - Excludes: drafts, restricted, secret")
    print("\n✅ Exclusion pattern")


def demonstrate_hybrid_scoring():
    """Demonstrate hybrid scoring with vector and keyword boost."""
    print("\n" + "=" * 60)
    print("4. HYBRID SCORING: Vector + keyword relevance")
    print("=" * 60)

    fb = ElasticsearchFilterBuilder
    query_vector = [0.1] * 384

    query = {
        "size": 10,
        "query": {
            "bool": {
                "must": [fb.knn_query("embedding", query_vector, k=10)],
                "should": [
                    {"match": {"title": {"query": "machine learning", "boost": 2.0}}},
                    {
                        "match": {
                            "abstract": {"query": "machine learning", "boost": 1.5}
                        }
                    },
                ],
                "filter": [
                    fb.term_filter("status", "published"),
                    fb.range_filter("year", gte=2023),
                ],
                "minimum_should_match": 0,
            }
        },
    }

    print("\n  Query structure:")
    print("  - Vector search (required)")
    print("  - Keyword boost on title (2x) and abstract (1.5x)")
    print("  - Filtered to published, recent documents")
    print("\n✅ Hybrid scoring query")


def demonstrate_aggregation_with_filtering():
    """Demonstrate filtered aggregations for faceted search."""
    print("\n" + "=" * 60)
    print("5. FILTERED AGGREGATIONS: Faceted search")
    print("=" * 60)

    query = {
        "size": 0,  # Only return aggregations
        "query": {
            "bool": {
                "filter": [
                    {"term": {"status": "published"}},
                    {"range": {"year": {"gte": 2023}}},
                ]
            }
        },
        "aggs": {
            "by_department": {"terms": {"field": "department", "size": 10}},
            "by_source_type": {"terms": {"field": "source_type", "size": 10}},
            "avg_credibility": {"avg": {"field": "credibility_score"}},
        },
    }

    print("\n  Aggregation structure:")
    print("  - Group by department")
    print("  - Group by source type")
    print("  - Average credibility score")
    print("  - Filters applied: published, recent")
    print("\n✅ Filtered aggregations for faceted navigation")


def demonstrate_practical_query():
    """Demonstrate a complete practical query."""
    print("\n" + "=" * 60)
    print("6. PRODUCTION QUERY: Comprehensive filtering")
    print("=" * 60)

    fb = ElasticsearchFilterBuilder
    query_vector = [0.1] * 384

    query = fb.build_search_query(
        query_vector=query_vector,
        filters=[
            fb.terms_filter("source_type", ["peer_reviewed", "academic", "official"]),
            fb.term_filter("source_verified", True),
            fb.range_filter("citation_count", gte=5),
            fb.range_filter("credibility_score", gte=0.7),
            fb.range_filter("publication_date", gte="2023-01-01"),
            fb.term_filter("language", "en"),
        ],
        must_not=[
            fb.term_filter("status", "retracted"),
            fb.terms_filter("classification", ["restricted", "secret"]),
        ],
        size=20,
        source_fields=["title", "abstract", "source", "credibility_score", "year"],
    )

    print("\n  Complete production query:")
    print("  - Verified academic/official sources")
    print("  - Quality: ≥5 citations, ≥0.7 credibility")
    print("  - Recent: 2023+")
    print("  - English language")
    print("  - Excludes retracted and classified")
    print("  - Returns title, abstract, source, score, year")
    print("\n✅ Production-ready query")


def main():
    """Run all Elasticsearch filtering demonstrations."""
    print("=" * 60)
    print("ELASTICSEARCH VECTOR + FILTERING DEMONSTRATION")
    print("=" * 60)
    print("\nNote: Demonstrates query construction patterns.")
    print("No actual Elasticsearch connection required.\n")

    demonstrate_verified_sources()
    demonstrate_date_range_filtering()
    demonstrate_exclusion_patterns()
    demonstrate_hybrid_scoring()
    demonstrate_aggregation_with_filtering()
    demonstrate_practical_query()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Elasticsearch combines traditional search features")
    print("   with vector similarity for powerful hybrid queries.")


if __name__ == "__main__":
    main()
