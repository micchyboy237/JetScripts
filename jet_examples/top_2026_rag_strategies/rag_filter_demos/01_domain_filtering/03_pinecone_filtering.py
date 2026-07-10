"""
03_pinecone_filtering.py

Pinecone vector database filtering examples.
Demonstrates cloud-native vector search with metadata filtering.
"""

from typing import Any, Dict, List


class PineconeFilterBuilder:
    """
    Filter builder for Pinecone's query language.
    Pinecone uses MongoDB-style query syntax.
    """

    @staticmethod
    def eq(field: str, value: Any) -> Dict:
        """Equal to filter."""
        return {field: {"$eq": value}}

    @staticmethod
    def ne(field: str, value: Any) -> Dict:
        """Not equal to filter."""
        return {field: {"$ne": value}}

    @staticmethod
    def in_list(field: str, values: List[Any]) -> Dict:
        """In list filter."""
        return {field: {"$in": values}}

    @staticmethod
    def nin(field: str, values: List[Any]) -> Dict:
        """Not in list filter."""
        return {field: {"$nin": values}}

    @staticmethod
    def gt(field: str, value: Any) -> Dict:
        """Greater than filter."""
        return {field: {"$gt": value}}

    @staticmethod
    def gte(field: str, value: Any) -> Dict:
        """Greater than or equal filter."""
        return {field: {"$gte": value}}

    @staticmethod
    def lt(field: str, value: Any) -> Dict:
        """Less than filter."""
        return {field: {"$lt": value}}

    @staticmethod
    def lte(field: str, value: Any) -> Dict:
        """Less than or equal filter."""
        return {field: {"$lte": value}}

    @staticmethod
    def combine_and(*filters: Dict) -> Dict:
        """Combine filters with AND logic."""
        valid = [f for f in filters if f]
        if not valid:
            return {}
        if len(valid) == 1:
            return valid[0]
        return {"$and": valid}

    @staticmethod
    def combine_or(*filters: Dict) -> Dict:
        """Combine filters with OR logic."""
        valid = [f for f in filters if f]
        if not valid:
            return {}
        if len(valid) == 1:
            return valid[0]
        return {"$or": valid}


def demonstrate_pinecone_filters():
    """
    Demonstrate Pinecone filter construction.
    Note: This runs without actual Pinecone connection.
    """
    fb = PineconeFilterBuilder

    print("\n" + "=" * 60)
    print("1. SIMPLE INCLUSION: Technology department")
    print("=" * 60)

    filter_dict = fb.eq("department", "Technology")
    print(f"\n  Filter: {filter_dict}")
    print("  Query equivalent:")
    print('    filter={"department": {"$eq": "Technology"}}')
    print("\n  ✅ Single field equality filter")

    print("\n" + "=" * 60)
    print("2. EXCLUSION: Not archived, year >= 2023")
    print("=" * 60)

    filter_dict = fb.combine_and(fb.ne("department", "Archived"), fb.gte("year", 2023))
    print(f"\n  Filter: {filter_dict}")
    print("  Excludes archived documents")
    print("  Only includes documents from 2023 onward")
    print("\n  ✅ Combined exclusion with range filter")

    print("\n" + "=" * 60)
    print("3. COMPLEX: Published + (Technology or Product) + recent")
    print("=" * 60)

    filter_dict = fb.combine_and(
        fb.eq("status", "published"),
        fb.combine_or(
            fb.eq("department", "Technology"), fb.eq("department", "Product")
        ),
        fb.gte("year", 2024),
    )
    print(f"\n  Filter: {filter_dict}")
    print("  Demonstrates nested AND/OR logic")
    print("\n  ✅ Complex multi-condition filter")

    print("\n" + "=" * 60)
    print("4. METADATA INCLUSION: Filter with field selection")
    print("=" * 60)

    # Pinecone allows specifying which metadata fields to return
    query_params = {
        "filter": fb.combine_and(
            fb.in_list("department", ["Technology", "Engineering"]),
            fb.gte("credibility_score", 0.8),
        ),
        "include_metadata": True,
        "include_values": False,
    }
    print(f"\n  Query params: {query_params}")
    print("  Returns only specified metadata fields")
    print("\n  ✅ Optimized metadata retrieval")

    print("\n" + "=" * 60)
    print("5. PAGINATION: Filtered query with limit and offset")
    print("=" * 60)

    query_params = {
        "filter": fb.combine_and(
            fb.eq("department", "Technology"), fb.gte("year", 2023)
        ),
        "top_k": 10,
        "include_metadata": True,
    }
    print(f"\n  Query params: {filter_dict}")
    print("  Returns top 10 matching documents")
    print("\n  ✅ Filtered pagination")


def demonstrate_pinecone_patterns():
    """Demonstrate common Pinecone filtering patterns."""
    fb = PineconeFilterBuilder

    print("\n" + "=" * 60)
    print("COMMON PINECONE PATTERNS")
    print("=" * 60)

    patterns = {
        "Date Range": fb.combine_and(
            fb.gte("created_at", "2024-01-01"), fb.lte("created_at", "2024-12-31")
        ),
        "Multi-Category": fb.in_list("category", ["report", "analysis", "whitepaper"]),
        "Quality Threshold": fb.combine_and(
            fb.gte("credibility_score", 0.7), fb.gte("citation_count", 5)
        ),
        "Security Level": fb.lte("clearance_level", "confidential"),
        "Exclude Drafts": fb.ne("status", "draft"),
        "Geo Filter": fb.in_list("region", ["US", "EU", "APAC"]),
    }

    for name, filter_dict in patterns.items():
        print(f"\n  {name}:")
        print(f"    {filter_dict}")


def main():
    """Run all Pinecone filtering demonstrations."""
    print("=" * 60)
    print("PINECONE FILTERING DEMONSTRATION")
    print("=" * 60)
    print("\nNote: Demonstrates filter construction patterns.")
    print("No actual Pinecone connection required.\n")

    demonstrate_pinecone_filters()
    demonstrate_pinecone_patterns()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Pinecone uses MongoDB-style query syntax for metadata filtering.")
    print("   Key operators: $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte, $and, $or")


if __name__ == "__main__":
    main()
