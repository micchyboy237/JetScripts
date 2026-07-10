"""
02_weaviate_advanced.py

Advanced Weaviate filtering with GraphQL-like composition.
Demonstrates complex filter composition using Weaviate v4 syntax.
"""

from enum import Enum
from typing import Any, List


class WeaviateOperator(str, Enum):
    """Weaviate filter operators."""

    EQUAL = "Equal"
    NOT_EQUAL = "NotEqual"
    GREATER_THAN = "GreaterThan"
    GREATER_OR_EQUAL = "GreaterThanEqual"
    LESS_THAN = "LessThan"
    LESS_OR_EQUAL = "LessThanEqual"
    LIKE = "Like"
    IS_NULL = "IsNull"
    CONTAINS_ANY = "ContainsAny"
    CONTAINS_ALL = "ContainsAll"


class WeaviateFilter:
    """
    Programmatic Weaviate filter builder.
    Generates Weaviate v4 Python client filter syntax.
    """

    def __init__(self):
        self.conditions = []

    @staticmethod
    def by_property(name: str) -> "PropertyFilter":
        """Start a property filter."""
        return PropertyFilter(name)

    @staticmethod
    def and_combine(*filters) -> str:
        """Combine filters with AND."""
        valid = [f for f in filters if f]
        if not valid:
            return ""
        if len(valid) == 1:
            return valid[0]
        return " & ".join(f"({f})" for f in valid)

    @staticmethod
    def or_combine(*filters) -> str:
        """Combine filters with OR."""
        valid = [f for f in filters if f]
        if not valid:
            return ""
        if len(valid) == 1:
            return valid[0]
        return " | ".join(f"({f})" for f in valid)


class PropertyFilter:
    """Filter for a specific property."""

    def __init__(self, property_name: str):
        self.property_name = property_name
        self.filter_parts = []

    def equal(self, value: Any) -> str:
        """Equal to filter."""
        if isinstance(value, str):
            return f'Filter.by_property("{self.property_name}").equal("{value}")'
        return f'Filter.by_property("{self.property_name}").equal({value})'

    def not_equal(self, value: Any) -> str:
        """Not equal to filter."""
        if isinstance(value, str):
            return f'Filter.by_property("{self.property_name}").not_equal("{value}")'
        return f'Filter.by_property("{self.property_name}").not_equal({value})'

    def greater_than(self, value: Any) -> str:
        """Greater than filter."""
        return f'Filter.by_property("{self.property_name}").greater_than({value})'

    def greater_or_equal(self, value: Any) -> str:
        """Greater than or equal filter."""
        return f'Filter.by_property("{self.property_name}").greater_or_equal({value})'

    def less_than(self, value: Any) -> str:
        """Less than filter."""
        return f'Filter.by_property("{self.property_name}").less_than({value})'

    def less_or_equal(self, value: Any) -> str:
        """Less than or equal filter."""
        return f'Filter.by_property("{self.property_name}").less_or_equal({value})'

    def like(self, pattern: str) -> str:
        """Pattern match filter."""
        return f'Filter.by_property("{self.property_name}").like("{pattern}")'

    def is_null(self, should_be_null: bool = True) -> str:
        """Null check filter."""
        return f'Filter.by_property("{self.property_name}").is_null({str(should_be_null).lower()})'

    def contains_any(self, values: List[Any]) -> str:
        """Contains any of the values."""
        vals = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in values)
        return f'Filter.by_property("{self.property_name}").contains_any([{vals}])'

    def contains_all(self, values: List[Any]) -> str:
        """Contains all of the values."""
        vals = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in values)
        return f'Filter.by_property("{self.property_name}").contains_all([{vals}])'


def demonstrate_simple_filters():
    """Demonstrate simple Weaviate filter construction."""
    print("\n" + "=" * 60)
    print("1. SIMPLE PROPERTY FILTERS")
    print("=" * 60)

    wf = WeaviateFilter

    filters = {
        "Exact match": wf.by_property("department").equal("Technology"),
        "Not equal": wf.by_property("status").not_equal("draft"),
        "Greater than": wf.by_property("year").greater_than(2022),
        "Range": wf.and_combine(
            wf.by_property("year").greater_or_equal(2023),
            wf.by_property("year").less_or_equal(2024),
        ),
        "Pattern match": wf.by_property("topic").like("%AI%"),
        "Null check": wf.by_property("deleted_at").is_null(True),
        "Contains any": wf.by_property("tags").contains_any(["AI", "ML", "cloud"]),
        "Contains all": wf.by_property("categories").contains_all(
            ["published", "verified"]
        ),
    }

    for name, filter_expr in filters.items():
        print(f"\n  {name}:")
        print(f"    {filter_expr}")

    print("\n✅ Simple filters demonstrated")


def demonstrate_complex_composition():
    """Demonstrate complex filter composition."""
    print("\n" + "=" * 60)
    print("2. COMPLEX FILTER COMPOSITION")
    print("=" * 60)

    wf = WeaviateFilter

    # Complex filter: (Tech OR Engineering) AND year >= 2023 AND topic like AI/ML
    complex_filter = wf.and_combine(
        wf.or_combine(
            wf.by_property("department").equal("Technology"),
            wf.by_property("department").equal("Engineering"),
        ),
        wf.by_property("year").greater_or_equal(2023),
        wf.or_combine(
            wf.by_property("topic").like("%AI%"),
            wf.by_property("topic").like("%ML%"),
            wf.by_property("topic").like("%machine learning%"),
        ),
    )

    print("\n  Complex filter:")
    print(f"  filters = {complex_filter}")

    print("\n  Equivalent hybrid query:")
    print("  response = client.collections.get('Documents').query.hybrid(")
    print("      query='machine learning advancements',")
    print("      filters=filters,")
    print("      limit=10")
    print("  )")

    print("\n✅ Complex composition demonstrated")


def demonstrate_hybrid_search_config():
    """Demonstrate hybrid search configuration."""
    print("\n" + "=" * 60)
    print("3. HYBRID SEARCH CONFIGURATION")
    print("=" * 60)

    configs = {
        "Pure vector (alpha=1.0)": {
            "alpha": 1.0,
            "description": "Only vector similarity, ignores keywords",
        },
        "Pure keyword (alpha=0.0)": {
            "alpha": 0.0,
            "description": "Only BM25 keyword search, ignores vectors",
        },
        "Balanced (alpha=0.5)": {
            "alpha": 0.5,
            "description": "Equal weighting of vector and keyword scores",
        },
        "Vector-biased (alpha=0.75)": {
            "alpha": 0.75,
            "description": "75% vector, 25% keyword - good for semantic search",
        },
        "Keyword-biased (alpha=0.25)": {
            "alpha": 0.25,
            "description": "25% vector, 75% keyword - good for exact matching",
        },
    }

    for name, config in configs.items():
        print(f"\n  {name}:")
        print(f"    Alpha: {config['alpha']}")
        print(f"    {config['description']}")

    print("\n✅ Hybrid search configurations explained")


def demonstrate_filter_with_hybrid_query():
    """Demonstrate complete hybrid query with filters."""
    print("\n" + "=" * 60)
    print("4. COMPLETE HYBRID QUERY")
    print("=" * 60)

    wf = WeaviateFilter

    # Build filters
    filters = wf.and_combine(
        wf.by_property("department").equal("Technology"),
        wf.by_property("year").greater_or_equal(2023),
        wf.by_property("status").equal("published"),
        wf.by_property("credibility_score").greater_or_equal(0.8),
        wf.or_combine(
            wf.by_property("topic").like("%AI%"), wf.by_property("topic").like("%ML%")
        ),
    )

    # Complete query code
    query_code = f"""
from weaviate.classes.query import Filter
import weaviate

client = weaviate.connect_to_local()

filters = {filters}

response = client.collections.get("Documents").query.hybrid(
    query="What are the latest AI advancements?",
    filters=filters,
    alpha=0.5,
    limit=10,
    return_properties=[
        "title", "content", "department", 
        "year", "credibility_score", "topic"
    ],
    autocut=2  # Auto-group results by relevance
)

for obj in response.objects:
    print(f"{{obj.properties['title']}}: {{obj.properties['credibility_score']}}")
"""

    print("\n  Complete query:")
    print(f"  {query_code}")

    print("\n✅ Complete hybrid query demonstrated")


def demonstrate_advanced_weaviate_patterns():
    """Demonstrate advanced Weaviate patterns."""
    print("\n" + "=" * 60)
    print("5. ADVANCED WEAVIATE PATTERNS")
    print("=" * 60)

    wf = WeaviateFilter

    patterns = {
        "Faceted Search": {
            "description": "Filter + aggregate by category",
            "filter": wf.by_property("year").greater_or_equal(2023),
        },
        "Semantic + Exact": {
            "description": "Vector search with exact metadata match",
            "filter": wf.and_combine(
                wf.by_property("department").equal("Technology"),
                wf.by_property("status").equal("published"),
            ),
        },
        "Time-Decayed Search": {
            "description": "Boost recent documents in results",
            "filter": wf.and_combine(
                wf.by_property("created_at").greater_or_equal("2023-01-01T00:00:00Z"),
                wf.by_property("status").not_equal("archived"),
            ),
        },
        "Multi-Collection Cross-Reference": {
            "description": "Filter based on related object properties",
            "filter": "Uses cross-references with MultiTargetFilter",
        },
    }

    for name, pattern in patterns.items():
        print(f"\n  {name}:")
        print(f"    {pattern['description']}")
        print(f"    {pattern['filter']}")

    print("\n✅ Advanced patterns documented")


def main():
    """Run all Weaviate advanced demonstrations."""
    print("=" * 60)
    print("WEAVIATE ADVANCED FILTERING")
    print("=" * 60)
    print("\nNote: Demonstrates Weaviate v4 filter syntax.")
    print("No actual Weaviate connection required.\n")

    demonstrate_simple_filters()
    demonstrate_complex_composition()
    demonstrate_hybrid_search_config()
    demonstrate_filter_with_hybrid_query()
    demonstrate_advanced_weaviate_patterns()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Weaviate v4's Python-native filter syntax makes")
    print("   complex queries readable and composable.")


if __name__ == "__main__":
    main()
