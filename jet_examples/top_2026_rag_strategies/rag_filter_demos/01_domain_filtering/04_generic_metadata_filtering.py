"""
04_generic_metadata_filtering.py

Basic metadata filtering concepts without external vector databases.
Demonstrates foundational patterns: simple filters, combining filters,
and building filter pipelines manually.

This is a NEW basic example not in the original markdown.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Document:
    """Simple document class for demonstration."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_sample_documents() -> List[Document]:
    """Create sample documents with metadata."""
    return [
        Document(
            "AI research paper on transformers",
            {
                "department": "Technology",
                "year": 2024,
                "doc_type": "research_paper",
                "credibility_score": 0.95,
                "status": "published",
            },
        ),
        Document(
            "Machine learning deployment guide",
            {
                "department": "Technology",
                "year": 2023,
                "doc_type": "guide",
                "credibility_score": 0.85,
                "status": "published",
            },
        ),
        Document(
            "HR policy updates for remote work",
            {
                "department": "HR",
                "year": 2024,
                "doc_type": "policy",
                "credibility_score": 0.70,
                "status": "published",
            },
        ),
        Document(
            "Employee benefits summary",
            {
                "department": "HR",
                "year": 2023,
                "doc_type": "summary",
                "credibility_score": 0.60,
                "status": "draft",
            },
        ),
        Document(
            "Q4 financial report",
            {
                "department": "Finance",
                "year": 2024,
                "doc_type": "report",
                "credibility_score": 0.90,
                "status": "published",
            },
        ),
        Document(
            "Budget planning template",
            {
                "department": "Finance",
                "year": 2023,
                "doc_type": "template",
                "credibility_score": 0.65,
                "status": "published",
            },
        ),
        Document(
            "Cloud migration strategy",
            {
                "department": "Technology",
                "year": 2024,
                "doc_type": "strategy",
                "credibility_score": 0.88,
                "status": "draft",
            },
        ),
        Document(
            "Annual compliance review",
            {
                "department": "Legal",
                "year": 2023,
                "doc_type": "review",
                "credibility_score": 0.92,
                "status": "published",
            },
        ),
    ]


class BasicFilter:
    """
    Basic filter operators for metadata filtering.
    This demonstrates the fundamental concepts that vector databases implement.
    """

    @staticmethod
    def eq(value: Any) -> callable:
        """Equal to filter."""
        return lambda x: x == value

    @staticmethod
    def ne(value: Any) -> callable:
        """Not equal to filter."""
        return lambda x: x != value

    @staticmethod
    def in_list(values: List[Any]) -> callable:
        """In list filter."""
        return lambda x: x in values

    @staticmethod
    def not_in(values: List[Any]) -> callable:
        """Not in list filter."""
        return lambda x: x not in values

    @staticmethod
    def gte(value: Any) -> callable:
        """Greater than or equal filter."""
        return lambda x: x >= value

    @staticmethod
    def lte(value: Any) -> callable:
        """Less than or equal filter."""
        return lambda x: x <= value

    @staticmethod
    def between(min_val: Any, max_val: Any) -> callable:
        """Between range filter."""
        return lambda x: min_val <= x <= max_val


def apply_filter(
    documents: List[Document], field: str, filter_func: callable
) -> List[Document]:
    """Apply a single filter to documents."""
    return [
        doc
        for doc in documents
        if field in doc.metadata and filter_func(doc.metadata[field])
    ]


def apply_and_filters(
    documents: List[Document], filters: Dict[str, callable]
) -> List[Document]:
    """Apply multiple filters with AND logic."""
    result = documents
    for field, filter_func in filters.items():
        result = apply_filter(result, field, filter_func)
    return result


def apply_or_filters(
    documents: List[Document], filters: List[Dict[str, callable]]
) -> List[Document]:
    """Apply multiple filter groups with OR logic."""
    results = set()
    for filter_group in filters:
        matching = apply_and_filters(documents, filter_group)
        results.update(matching)
    return list(results)


def demonstrate_basic_inclusion():
    """Demonstrate basic inclusion filtering."""
    print("\n" + "=" * 60)
    print("1. BASIC INCLUSION: Technology department only")
    print("=" * 60)

    docs = create_sample_documents()
    filtered = apply_filter(docs, "department", BasicFilter.eq("Technology"))

    for doc in filtered:
        print(f"  [{doc.metadata['doc_type']}] {doc.content}")
    print(f"\n✅ {len(filtered)}/{len(docs)} documents match")


def demonstrate_basic_exclusion():
    """Demonstrate basic exclusion filtering."""
    print("\n" + "=" * 60)
    print("2. BASIC EXCLUSION: Exclude drafts")
    print("=" * 60)

    docs = create_sample_documents()
    filtered = apply_filter(docs, "status", BasicFilter.ne("draft"))

    for doc in filtered:
        status = doc.metadata["status"]
        print(f"  [{status}] {doc.content}")
    print(f"\n✅ {len(filtered)}/{len(docs)} documents are not drafts")


def demonstrate_multiple_conditions():
    """Demonstrate filtering with multiple conditions (AND)."""
    print("\n" + "=" * 60)
    print("3. MULTIPLE CONDITIONS (AND): Technology + 2024 + published")
    print("=" * 60)

    docs = create_sample_documents()
    filters = {
        "department": BasicFilter.eq("Technology"),
        "year": BasicFilter.eq(2024),
        "status": BasicFilter.eq("published"),
    }
    filtered = apply_and_filters(docs, filters)

    for doc in filtered:
        print(
            f"  [{doc.metadata['department']}] [{doc.metadata['year']}] {doc.content}"
        )
    print(f"\n✅ {len(filtered)}/{len(docs)} documents match all conditions")


def demonstrate_or_logic():
    """Demonstrate OR logic across filter groups."""
    print("\n" + "=" * 60)
    print("4. OR LOGIC: Technology OR Finance, published only")
    print("=" * 60)

    docs = create_sample_documents()
    filter_groups = [
        {
            "department": BasicFilter.eq("Technology"),
            "status": BasicFilter.eq("published"),
        },
        {
            "department": BasicFilter.eq("Finance"),
            "status": BasicFilter.eq("published"),
        },
    ]
    filtered = apply_or_filters(docs, filter_groups)

    for doc in filtered:
        print(f"  [{doc.metadata['department']}] {doc.content}")
    print(f"\n✅ {len(filtered)}/{len(docs)} documents match")


def demonstrate_range_filter():
    """Demonstrate range filtering with credibility scores."""
    print("\n" + "=" * 60)
    print("5. RANGE FILTER: Credibility score >= 0.85")
    print("=" * 60)

    docs = create_sample_documents()
    filtered = apply_filter(docs, "credibility_score", BasicFilter.gte(0.85))

    for doc in filtered:
        print(f"  [Score: {doc.metadata['credibility_score']}] {doc.content}")
    print(f"\n✅ {len(filtered)}/{len(docs)} documents meet quality threshold")


def demonstrate_combined_complex():
    """Demonstrate complex combined filtering."""
    print("\n" + "=" * 60)
    print(
        "6. COMPLEX COMBINED: Published + (Tech or Finance) + year 2024 + quality >= 0.8"
    )
    print("=" * 60)

    docs = create_sample_documents()

    # First, get published documents from Tech or Finance in 2024
    filter_groups = [
        {"department": BasicFilter.eq("Technology"), "year": BasicFilter.eq(2024)},
        {"department": BasicFilter.eq("Finance"), "year": BasicFilter.eq(2024)},
    ]
    filtered = apply_or_filters(docs, filter_groups)

    # Then apply quality and status filters
    filtered = apply_and_filters(
        filtered,
        {
            "status": BasicFilter.eq("published"),
            "credibility_score": BasicFilter.gte(0.8),
        },
    )

    for doc in filtered:
        print(
            f"  [{doc.metadata['department']}] [{doc.metadata['year']}] "
            f"[Score: {doc.metadata['credibility_score']}] {doc.content}"
        )
    print(f"\n✅ {len(filtered)}/{len(docs)} documents match all criteria")


def main():
    """Run all basic filtering demonstrations."""
    print("=" * 60)
    print("BASIC METADATA FILTERING CONCEPTS")
    print("=" * 60)
    print("\nThis demonstrates fundamental filtering patterns")
    print("that vector databases implement internally.\n")

    demonstrate_basic_inclusion()
    demonstrate_basic_exclusion()
    demonstrate_multiple_conditions()
    demonstrate_or_logic()
    demonstrate_range_filter()
    demonstrate_combined_complex()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 These patterns form the foundation of all vector database filtering.")


if __name__ == "__main__":
    main()
