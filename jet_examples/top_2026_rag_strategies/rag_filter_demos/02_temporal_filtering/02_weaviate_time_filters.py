"""
02_weaviate_time_filters.py

Weaviate temporal filtering with hybrid search.
Demonstrates combining vector search with time-based GraphQL filters.
"""

from datetime import datetime
from enum import Enum
from typing import Any


class FilterOperator(str, Enum):
    """Weaviate filter operators."""

    EQUAL = "Equal"
    NOT_EQUAL = "NotEqual"
    GREATER_THAN = "GreaterThan"
    GREATER_OR_EQUAL = "GreaterThanEqual"
    LESS_THAN = "LessThan"
    LESS_OR_EQUAL = "LessThanEqual"
    LIKE = "Like"
    WITHIN_GEO_RANGE = "WithinGeoRange"
    AND = "And"
    OR = "Or"


class WeaviateFilterBuilder:
    """
    Filter builder for Weaviate's GraphQL filter syntax.
    Weaviate v4 uses Python-native filter composition.
    """

    @staticmethod
    def equal(field: str, value: Any) -> str:
        """Equal filter."""
        return f'Filter.by_property("{field}").equal("{value}")'

    @staticmethod
    def not_equal(field: str, value: Any) -> str:
        """Not equal filter."""
        return f'Filter.by_property("{field}").not_equal("{value}")'

    @staticmethod
    def greater_than(field: str, value: Any) -> str:
        """Greater than filter."""
        return f'Filter.by_property("{field}").greater_than("{value}")'

    @staticmethod
    def greater_or_equal(field: str, value: Any) -> str:
        """Greater than or equal filter."""
        return f'Filter.by_property("{field}").greater_or_equal({value})'

    @staticmethod
    def less_than(field: str, value: Any) -> str:
        """Less than filter."""
        return f'Filter.by_property("{field}").less_than("{value}")'

    @staticmethod
    def less_or_equal(field: str, value: Any) -> str:
        """Less than or equal filter."""
        return f'Filter.by_property("{field}").less_or_equal({value})'

    @staticmethod
    def like(field: str, pattern: str) -> str:
        """Like (pattern match) filter."""
        return f'Filter.by_property("{field}").like("{pattern}")'

    @staticmethod
    def and_combine(*filters: str) -> str:
        """Combine filters with AND."""
        if len(filters) == 1:
            return filters[0]
        combined = " & ".join(f"({f})" for f in filters)
        return combined

    @staticmethod
    def or_combine(*filters: str) -> str:
        """Combine filters with OR."""
        if len(filters) == 1:
            return filters[0]
        combined = " | ".join(f"({f})" for f in filters)
        return combined


def demonstrate_year_filtering():
    """Demonstrate year-based filtering."""
    wb = WeaviateFilterBuilder

    print("\n" + "=" * 60)
    print("1. SPECIFIC YEAR RANGE: 2022-2024")
    print("=" * 60)

    filter_expr = wb.and_combine(
        wb.greater_or_equal("year", 2022), wb.less_or_equal("year", 2024)
    )

    print("\n  Python filter composition:")
    print(f"  filters = {filter_expr}")
    print("\n  Equivalent hybrid query:")
    print("  response = client.collections.get('Documents').query.hybrid(")
    print("      query='financial performance',")
    print("      filters=filters,")
    print("      limit=10")
    print("  )")
    print("\n  ✅ Year range filter")


def demonstrate_recency_filter():
    """Demonstrate recency-based filtering."""
    wb = WeaviateFilterBuilder
    current_year = datetime.now().year
    min_year = current_year - 2

    print("\n" + "=" * 60)
    print("2. RECENT DOCUMENTS: Last 2 years")
    print("=" * 60)

    filter_expr = wb.greater_or_equal("publication_date", f"{min_year}-01-01T00:00:00Z")

    print("\n  Python filter composition:")
    print(f"  filters = {filter_expr}")
    print("\n  This ensures only documents from the last 2 years are returned.")
    print(f"  Current year: {current_year}, minimum year: {min_year}")
    print("\n  ✅ Recency filter")


def demonstrate_complex_temporal():
    """Demonstrate complex temporal filtering combinations."""
    wb = WeaviateFilterBuilder

    print("\n" + "=" * 60)
    print("3. COMPLEX: Recent + specific departments + published")
    print("=" * 60)

    current_year = datetime.now().year

    filter_expr = wb.and_combine(
        wb.greater_or_equal("year", current_year - 1),
        wb.or_combine(
            wb.equal("department", "Technology"), wb.equal("department", "Engineering")
        ),
        wb.equal("status", "published"),
    )

    print("\n  Python filter composition:")
    print(f"  filters = {filter_expr}")
    print(f"\n  Logic: (year >= {current_year - 1}) AND")
    print("         (department = Technology OR department = Engineering) AND")
    print("         (status = published)")
    print("\n  ✅ Complex temporal filter")


def demonstrate_hybrid_search_patterns():
    """Demonstrate hybrid search with temporal filters."""
    wb = WeaviateFilterBuilder

    print("\n" + "=" * 60)
    print("4. HYBRID SEARCH: Vector + Keyword + Temporal")
    print("=" * 60)

    # Weaviate hybrid search combines BM25 and vector search
    query_template = """
    response = client.collections.get("{collection}").query.hybrid(
        query="{query_text}",
        filters={filters},
        alpha=0.5,  # Balance between vector and keyword search
        limit=10,
        return_properties=["title", "content", "year", "department"]
    )
    """

    filter_expr = wb.and_combine(
        wb.greater_or_equal("year", 2023), wb.like("topic", "%AI%")
    )

    print("\n  Query template:")
    print(
        f"  {query_template.format(collection='Documents', query_text='machine learning advancements', filters=filter_expr)}"
    )
    print("\n  Alpha parameter controls vector vs keyword balance:")
    print("  - alpha=0: Pure keyword search (BM25)")
    print("  - alpha=1: Pure vector search")
    print("  - alpha=0.5: Equal weighting")
    print("\n  ✅ Hybrid search pattern")


def demonstrate_advanced_temporal():
    """Demonstrate advanced temporal patterns."""
    wb = WeaviateFilterBuilder

    print("\n" + "=" * 60)
    print("ADVANCED TEMPORAL PATTERNS")
    print("=" * 60)

    patterns = {
        "Freshness Window": wb.and_combine(
            wb.greater_or_equal(
                "updated_at", f"{datetime.now().year - 1}-01-01T00:00:00Z"
            ),
            wb.equal("status", "active"),
        ),
        "Archival Detection": wb.and_combine(
            wb.less_than("created_at", f"{datetime.now().year - 3}-01-01T00:00:00Z"),
            wb.not_equal("status", "permanent"),
        ),
        "Quarterly Filter": wb.and_combine(
            wb.greater_or_equal("quarter", "Q1"),
            wb.less_or_equal("quarter", "Q2"),
            wb.equal("year", datetime.now().year),
        ),
        "Compliance Timeline": wb.and_combine(
            wb.greater_or_equal("effective_date", "2020-01-01T00:00:00Z"),
            wb.less_or_equal("expiry_date", "2025-12-31T23:59:59Z"),
            wb.equal("compliance_status", "active"),
        ),
    }

    for name, filter_expr in patterns.items():
        print(f"\n  {name}:")
        print(f"    {filter_expr}")


def main():
    """Run all Weaviate temporal filtering demonstrations."""
    print("=" * 60)
    print("WEAVIATE TEMPORAL FILTERING DEMONSTRATION")
    print("=" * 60)
    print("\nNote: Demonstrates filter construction patterns.")
    print("No actual Weaviate connection required.\n")

    demonstrate_year_filtering()
    demonstrate_recency_filter()
    demonstrate_complex_temporal()
    demonstrate_hybrid_search_patterns()
    demonstrate_advanced_temporal()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Weaviate v4 uses Python-native filter composition")
    print("   with & (AND) and | (OR) operators for readability.")


if __name__ == "__main__":
    main()
