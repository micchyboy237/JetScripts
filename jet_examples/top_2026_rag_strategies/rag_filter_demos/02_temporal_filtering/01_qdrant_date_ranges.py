"""
01_qdrant_date_ranges.py

Qdrant temporal filtering with date ranges.
Demonstrates high-performance vector search with time-based constraints.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List


class QdrantFilterBuilder:
    """
    Filter builder for Qdrant's filter system.
    Qdrant uses a structured filter model with must/should/must_not conditions.
    """

    @staticmethod
    def match(field: str, value: Any) -> Dict:
        """Exact match filter."""
        return {"key": field, "match": {"value": value}}

    @staticmethod
    def match_any(field: str, values: List[Any]) -> Dict:
        """Match any of the values."""
        return {"key": field, "match": {"any": values}}

    @staticmethod
    def range_filter(
        field: str, gt: Any = None, gte: Any = None, lt: Any = None, lte: Any = None
    ) -> Dict:
        """Range filter with optional bounds."""
        range_dict = {}
        if gt is not None:
            range_dict["gt"] = gt
        if gte is not None:
            range_dict["gte"] = gte
        if lt is not None:
            range_dict["lt"] = lt
        if lte is not None:
            range_dict["lte"] = lte

        return {"key": field, "range": range_dict}

    @staticmethod
    def date_after(field: str, date: datetime) -> Dict:
        """Filter dates after a specific date."""
        return QdrantFilterBuilder.range_filter(field, gt=date.timestamp())

    @staticmethod
    def date_before(field: str, date: datetime) -> Dict:
        """Filter dates before a specific date."""
        return QdrantFilterBuilder.range_filter(field, lt=date.timestamp())

    @staticmethod
    def date_between(field: str, start: datetime, end: datetime) -> Dict:
        """Filter dates between two dates."""
        return QdrantFilterBuilder.range_filter(
            field, gte=start.timestamp(), lte=end.timestamp()
        )

    @staticmethod
    def within_last_days(field: str, days: int) -> Dict:
        """Filter within the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return QdrantFilterBuilder.range_filter(field, gte=cutoff.timestamp())

    @staticmethod
    def build_filter(
        must: List[Dict] = None, should: List[Dict] = None, must_not: List[Dict] = None
    ) -> Dict:
        """Build a complete Qdrant filter."""
        filter_dict = {}
        if must:
            filter_dict["must"] = must
        if should:
            filter_dict["should"] = should
        if must_not:
            filter_dict["must_not"] = must_not
        return filter_dict


def demonstrate_basic_date_filters():
    """Demonstrate basic date filtering patterns."""
    fb = QdrantFilterBuilder

    print("\n" + "=" * 60)
    print("1. RECENT DOCUMENTS: Last 30 days")
    print("=" * 60)

    filter_dict = fb.build_filter(must=[fb.within_last_days("created_at", 30)])
    print("\n  Filter structure:")
    print(f"  {filter_dict}")
    print("\n  ✅ Returns documents from last 30 days")

    print("\n" + "=" * 60)
    print("2. SPECIFIC YEAR: 2024 only")
    print("=" * 60)

    start_2024 = datetime(2024, 1, 1)
    end_2024 = datetime(2024, 12, 31, 23, 59, 59)

    filter_dict = fb.build_filter(
        must=[fb.date_between("publish_date", start_2024, end_2024)]
    )
    print("\n  Filter structure:")
    print(f"  {filter_dict}")
    print("\n  ✅ Returns documents from 2024")


def demonstrate_combined_temporal():
    """Demonstrate combining temporal filters with other criteria."""
    fb = QdrantFilterBuilder

    print("\n" + "=" * 60)
    print("3. COMBINED: Last 90 days + published + technology")
    print("=" * 60)

    filter_dict = fb.build_filter(
        must=[
            fb.within_last_days("created_at", 90),
            fb.match("status", "published"),
            fb.match("department", "Technology"),
        ]
    )
    print("\n  Filter structure:")
    print(f"  {filter_dict}")
    print("\n  ✅ Recent published technology documents")

    print("\n" + "=" * 60)
    print("4. EXCLUSION: Not older than 2 years, not archived")
    print("=" * 60)

    two_years_ago = datetime.now() - timedelta(days=730)

    filter_dict = fb.build_filter(
        must=[fb.date_after("created_at", two_years_ago)],
        must_not=[fb.match("status", "archived")],
    )
    print("\n  Filter structure:")
    print(f"  {filter_dict}")
    print("\n  ✅ Recent non-archived documents")


def demonstrate_expiry_patterns():
    """Demonstrate document expiry and archival patterns."""
    fb = QdrantFilterBuilder

    print("\n" + "=" * 60)
    print("5. EXPIRY DETECTION: Documents older than 1 year")
    print("=" * 60)

    one_year_ago = datetime.now() - timedelta(days=365)

    filter_dict = fb.build_filter(must=[fb.date_before("created_at", one_year_ago)])
    print("\n  Filter structure:")
    print(f"  {filter_dict}")
    print("\n  ✅ Identifies documents ready for archival")

    print("\n" + "=" * 60)
    print("6. RETENTION WINDOW: 90 days to 2 years old")
    print("=" * 60)

    two_years_ago = datetime.now() - timedelta(days=730)
    ninety_days_ago = datetime.now() - timedelta(days=90)

    filter_dict = fb.build_filter(
        must=[fb.date_between("created_at", two_years_ago, ninety_days_ago)]
    )
    print("\n  Filter structure:")
    print(f"  {filter_dict}")
    print("\n  ✅ Documents in retention window")


def demonstrate_practical_scenarios():
    """Demonstrate practical temporal filtering scenarios."""
    fb = QdrantFilterBuilder

    print("\n" + "=" * 60)
    print("PRACTICAL TEMPORAL SCENARIOS")
    print("=" * 60)

    scenarios = {
        "Monthly Report": fb.build_filter(
            must=[fb.within_last_days("created_at", 30), fb.match("doc_type", "report")]
        ),
        "Quarterly Review": fb.build_filter(
            must=[
                fb.within_last_days("created_at", 90),
                fb.match_any("department", ["Finance", "Operations"]),
            ]
        ),
        "Annual Archive": fb.build_filter(
            must=[fb.date_before("created_at", datetime.now() - timedelta(days=365))],
            must_not=[fb.match("status", "permanent")],
        ),
        "Compliance Window": fb.build_filter(
            must=[
                fb.date_between("created_at", datetime(2020, 1, 1), datetime.now()),
                fb.match("doc_type", "compliance"),
            ]
        ),
    }

    for name, filter_dict in scenarios.items():
        print(f"\n  {name}:")
        print(f"    Conditions: {len(filter_dict.get('must', []))} required")
        if "must_not" in filter_dict:
            print(f"    Exclusions: {len(filter_dict['must_not'])} applied")


def main():
    """Run all Qdrant temporal filtering demonstrations."""
    print("=" * 60)
    print("QDRANT TEMPORAL FILTERING DEMONSTRATION")
    print("=" * 60)
    print("\nNote: Demonstrates filter construction patterns.")
    print("No actual Qdrant connection required.\n")

    demonstrate_basic_date_filters()
    demonstrate_combined_temporal()
    demonstrate_expiry_patterns()
    demonstrate_practical_scenarios()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Qdrant uses structured must/should/must_not conditions")
    print("   with Range objects for temporal filtering.")


if __name__ == "__main__":
    main()
