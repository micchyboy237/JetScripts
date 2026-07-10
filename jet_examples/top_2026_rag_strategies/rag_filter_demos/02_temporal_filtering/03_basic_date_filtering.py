"""
03_basic_date_filtering.py

Basic date and temporal filtering concepts without external dependencies.
Demonstrates foundational patterns for time-based document filtering.

This is a NEW basic example not in the original markdown.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List


@dataclass
class Document:
    """Simple document class with temporal metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_sample_documents() -> List[Document]:
    """Create sample documents with various dates."""
    now = datetime.now()

    return [
        Document(
            "Latest product roadmap",
            {
                "title": "Q3 2024 Product Roadmap",
                "created_at": now - timedelta(days=5),
                "year": now.year,
                "status": "active",
            },
        ),
        Document(
            "Annual strategy document",
            {
                "title": "2024 Annual Strategy",
                "created_at": now - timedelta(days=180),
                "year": now.year,
                "status": "active",
            },
        ),
        Document(
            "Archived project proposal",
            {
                "title": "Project Phoenix Proposal",
                "created_at": now - timedelta(days=400),
                "year": now.year - 1,
                "status": "archived",
            },
        ),
        Document(
            "Recent team update",
            {
                "title": "Weekly Team Update",
                "created_at": now - timedelta(days=1),
                "year": now.year,
                "status": "active",
            },
        ),
        Document(
            "Historical financial data",
            {
                "title": "Q1 2020 Financial Report",
                "created_at": now - timedelta(days=1200),
                "year": 2020,
                "status": "archived",
            },
        ),
        Document(
            "Compliance documentation",
            {
                "title": "GDPR Compliance 2023",
                "created_at": now - timedelta(days=300),
                "year": 2023,
                "status": "active",
            },
        ),
    ]


class TemporalFilter:
    """Temporal filtering operations."""

    @staticmethod
    def after_date(cutoff: datetime) -> callable:
        """Filter documents created after a specific date."""
        return lambda date: date > cutoff

    @staticmethod
    def before_date(cutoff: datetime) -> callable:
        """Filter documents created before a specific date."""
        return lambda date: date < cutoff

    @staticmethod
    def between_dates(start: datetime, end: datetime) -> callable:
        """Filter documents created between two dates."""
        return lambda date: start <= date <= end

    @staticmethod
    def within_last_days(days: int) -> callable:
        """Filter documents created within the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return lambda date: date >= cutoff

    @staticmethod
    def in_year(year: int) -> callable:
        """Filter documents from a specific year."""
        return lambda date: date.year == year

    @staticmethod
    def years_between(min_year: int, max_year: int) -> callable:
        """Filter documents from a range of years."""
        return lambda date: min_year <= date.year <= max_year

    @staticmethod
    def older_than_days(days: int) -> callable:
        """Filter documents older than N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return lambda date: date < cutoff

    @staticmethod
    def newer_than_days(days: int) -> callable:
        """Filter documents newer than N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return lambda date: date > cutoff


def apply_date_filter(
    documents: List[Document], date_field: str, filter_func: callable
) -> List[Document]:
    """Apply a date filter to documents."""
    return [
        doc
        for doc in documents
        if date_field in doc.metadata and filter_func(doc.metadata[date_field])
    ]


def demonstrate_recent_documents():
    """Demonstrate filtering for recent documents."""
    print("\n" + "=" * 60)
    print("1. RECENT DOCUMENTS: Within last 30 days")
    print("=" * 60)

    docs = create_sample_documents()
    filtered = apply_date_filter(
        docs, "created_at", TemporalFilter.within_last_days(30)
    )

    for doc in filtered:
        days_ago = (datetime.now() - doc.metadata["created_at"]).days
        print(f"  [{days_ago} days ago] {doc.metadata['title']}")
    print(f"\n✅ {len(filtered)}/{len(docs)} documents are recent")


def demonstrate_year_filter():
    """Demonstrate filtering by year."""
    print("\n" + "=" * 60)
    print("2. YEAR FILTER: Current year documents only")
    print("=" * 60)

    docs = create_sample_documents()
    current_year = datetime.now().year
    filtered = apply_date_filter(
        docs, "created_at", TemporalFilter.in_year(current_year)
    )

    for doc in filtered:
        print(f"  [{doc.metadata['year']}] {doc.metadata['title']}")
    print(f"\n✅ {len(filtered)}/{len(docs)} documents from {current_year}")


def demonstrate_date_range():
    """Demonstrate filtering by date range."""
    print("\n" + "=" * 60)
    print("3. DATE RANGE: Last 6 months but not last 7 days")
    print("=" * 60)

    docs = create_sample_documents()
    now = datetime.now()
    start_date = now - timedelta(days=180)
    end_date = now - timedelta(days=7)

    filtered = apply_date_filter(
        docs, "created_at", TemporalFilter.between_dates(start_date, end_date)
    )

    for doc in filtered:
        days_ago = (datetime.now() - doc.metadata["created_at"]).days
        print(f"  [{days_ago} days ago] {doc.metadata['title']}")
    print(f"\n✅ {len(filtered)}/{len(docs)} documents in range")


def demonstrate_exclude_old():
    """Demonstrate excluding old documents."""
    print("\n" + "=" * 60)
    print("4. EXCLUDE OLD: Not older than 1 year")
    print("=" * 60)

    docs = create_sample_documents()
    filtered = apply_date_filter(
        docs, "created_at", TemporalFilter.within_last_days(365)
    )

    for doc in filtered:
        days_ago = (datetime.now() - doc.metadata["created_at"]).days
        print(f"  [{days_ago} days ago] {doc.metadata['title']}")

    excluded = len(docs) - len(filtered)
    print(f"\n✅ {len(filtered)}/{len(docs)} documents kept, {excluded} excluded")


def demonstrate_expiry_check():
    """Demonstrate document expiry/archival filtering."""
    print("\n" + "=" * 60)
    print("5. EXPIRY CHECK: Documents for archival (> 1 year old)")
    print("=" * 60)

    docs = create_sample_documents()
    filtered = apply_date_filter(
        docs, "created_at", TemporalFilter.older_than_days(365)
    )

    for doc in filtered:
        days_ago = (datetime.now() - doc.metadata["created_at"]).days
        years = days_ago / 365
        print(f"  [{years:.1f} years old] {doc.metadata['title']}")
    print(f"\n✅ {len(filtered)}/{len(docs)} documents ready for archival")


def demonstrate_combined_temporal():
    """Demonstrate combining temporal filters with other criteria."""
    print("\n" + "=" * 60)
    print("6. COMBINED: Recent (30 days) + active status")
    print("=" * 60)

    docs = create_sample_documents()

    # Filter by recency
    recent_docs = apply_date_filter(
        docs, "created_at", TemporalFilter.within_last_days(30)
    )

    # Filter by status
    active_docs = [d for d in recent_docs if d.metadata.get("status") == "active"]

    for doc in active_docs:
        days_ago = (datetime.now() - doc.metadata["created_at"]).days
        print(
            f"  [{days_ago} days ago] [{doc.metadata['status']}] {doc.metadata['title']}"
        )
    print(f"\n✅ {len(active_docs)}/{len(docs)} documents match all criteria")


def main():
    """Run all basic temporal filtering demonstrations."""
    print("=" * 60)
    print("BASIC TEMPORAL FILTERING CONCEPTS")
    print("=" * 60)
    print("\nThis demonstrates fundamental date/time filtering patterns")
    print("that can be applied to any document store.\n")

    demonstrate_recent_documents()
    demonstrate_year_filter()
    demonstrate_date_range()
    demonstrate_exclude_old()
    demonstrate_expiry_check()
    demonstrate_combined_temporal()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Temporal filtering is essential for:")
    print("  - Ensuring information freshness")
    print("  - Archival and retention policies")
    print("  - Time-sensitive query responses")


if __name__ == "__main__":
    main()
