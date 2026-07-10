"""
01_hybrid_multicriteria.py

Advanced hybrid multi-criteria filtering patterns.
Demonstrates complex AND/OR logic for sophisticated filtering requirements.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class LogicOperator(str, Enum):
    """Logical operators for filter composition."""

    AND = "$and"
    OR = "$or"
    NOT = "$not"


class ComparisonOperator(str, Enum):
    """Comparison operators for field conditions."""

    EQ = "$eq"
    NE = "$ne"
    GT = "$gt"
    GTE = "$gte"
    LT = "$lt"
    LTE = "$lte"
    IN = "$in"
    NIN = "$nin"


@dataclass
class FilterCondition:
    """A single filter condition."""

    field: str
    operator: ComparisonOperator
    value: Any

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {self.field: {self.operator.value: self.value}}


@dataclass
class FilterGroup:
    """A group of filter conditions combined with AND/OR."""

    operator: LogicOperator
    conditions: List[Any] = field(
        default_factory=list
    )  # FilterCondition or FilterGroup

    def add_condition(self, field: str, operator: ComparisonOperator, value: Any):
        """Add a condition to this group."""
        self.conditions.append(FilterCondition(field, operator, value))
        return self

    def add_group(self, group: "FilterGroup"):
        """Add a nested filter group."""
        self.conditions.append(group)
        return self

    def to_dict(self) -> Dict:
        """Convert to nested dictionary format."""
        if len(self.conditions) == 1:
            cond = self.conditions[0]
            return (
                cond.to_dict() if isinstance(cond, FilterCondition) else cond.to_dict()
            )

        return {
            self.operator.value: [
                c.to_dict() if hasattr(c, "to_dict") else c for c in self.conditions
            ]
        }


class HybridFilterBuilder:
    """
    Build complex hybrid filters with nested AND/OR logic.

    Example complex filter:
    (Department = Tech OR Department = Engineering)
    AND (Year >= 2023)
    AND (Topic = AI OR Topic = ML)
    AND (Status != draft)
    AND (Credibility >= 0.7)
    """

    @staticmethod
    def build_department_tech_or_engineering() -> FilterGroup:
        """Build department filter: Technology OR Engineering."""
        group = FilterGroup(LogicOperator.OR)
        group.add_condition("department", ComparisonOperator.EQ, "Technology")
        group.add_condition("department", ComparisonOperator.EQ, "Engineering")
        return group

    @staticmethod
    def build_topic_ai_or_ml() -> FilterGroup:
        """Build topic filter: AI OR ML."""
        group = FilterGroup(LogicOperator.OR)
        group.add_condition("topic", ComparisonOperator.EQ, "Artificial Intelligence")
        group.add_condition("topic", ComparisonOperator.EQ, "Machine Learning")
        group.add_condition("topic", ComparisonOperator.EQ, "AI")
        group.add_condition("topic", ComparisonOperator.EQ, "ML")
        return group

    @staticmethod
    def build_complex_production_filter() -> Dict:
        """Build the complex production filter from the documentation."""

        # Create outer AND group
        root = FilterGroup(LogicOperator.AND)

        # Condition 1: (Department = Tech OR Department = Engineering)
        root.add_group(HybridFilterBuilder.build_department_tech_or_engineering())

        # Condition 2: Year >= 2023
        root.add_condition("year", ComparisonOperator.GTE, 2023)

        # Condition 3: (Topic = AI OR Topic = ML OR ...)
        root.add_group(HybridFilterBuilder.build_topic_ai_or_ml())

        # Condition 4: Status != draft
        root.add_condition("status", ComparisonOperator.NE, "draft")

        # Condition 5: Confidence >= 0.7
        root.add_condition("confidence_score", ComparisonOperator.GTE, 0.7)

        return root.to_dict()

    @staticmethod
    def build_geo_compliance_filter(regions: List[str]) -> Dict:
        """Build geographic compliance filter."""
        root = FilterGroup(LogicOperator.AND)

        # Region restriction
        root.add_condition("region", ComparisonOperator.IN, regions)

        # Data sovereignty
        data_group = FilterGroup(LogicOperator.OR)
        data_group.add_condition("data_residency", ComparisonOperator.EQ, "local")
        data_group.add_condition("cross_border_approved", ComparisonOperator.EQ, True)
        root.add_group(data_group)

        # Compliance status
        root.add_condition("compliance_status", ComparisonOperator.EQ, "approved")

        return root.to_dict()

    @staticmethod
    def build_time_window_filter(
        start_date: str, end_date: str, include_drafts: bool = False
    ) -> Dict:
        """Build time window filter with optional draft inclusion."""
        root = FilterGroup(LogicOperator.AND)

        # Date range
        root.add_condition("publish_date", ComparisonOperator.GTE, start_date)
        root.add_condition("publish_date", ComparisonOperator.LTE, end_date)

        # Draft handling
        if not include_drafts:
            root.add_condition("status", ComparisonOperator.NE, "draft")

        return root.to_dict()


def demonstrate_nested_filter_building():
    """Demonstrate building nested AND/OR filters."""
    print("\n" + "=" * 60)
    print("1. NESTED FILTER CONSTRUCTION")
    print("=" * 60)

    # Build complex filter
    complex_filter = HybridFilterBuilder.build_complex_production_filter()

    print("\n  Complex production filter:")
    print(f"  {json.dumps(complex_filter, indent=4)}")

    print("\n  Filter logic:")
    print("  ┌─ AND")
    print("  │  ├─ OR: Department = Technology | Engineering")
    print("  │  ├─ Year >= 2023")
    print("  │  ├─ OR: Topic = AI | ML | Artificial Intelligence | Machine Learning")
    print("  │  ├─ Status != draft")
    print("  │  └─ Confidence >= 0.7")

    print("\n✅ Nested filter built successfully")


def demonstrate_geo_compliance():
    """Demonstrate geographic compliance filtering."""
    print("\n" + "=" * 60)
    print("2. GEOGRAPHIC COMPLIANCE FILTER")
    print("=" * 60)

    # EU compliance filter
    eu_filter = HybridFilterBuilder.build_geo_compliance_filter(["EU", "EEA"])

    print("\n  EU Compliance filter:")
    print(f"  {json.dumps(eu_filter, indent=4)}")

    # Global filter
    global_filter = HybridFilterBuilder.build_geo_compliance_filter(
        ["US", "EU", "APAC"]
    )

    print("\n  Global filter:")
    print(f"  {json.dumps(global_filter, indent=4)}")

    print("\n✅ Geographic filters built")


def demonstrate_time_windows():
    """Demonstrate time window filtering."""
    print("\n" + "=" * 60)
    print("3. TIME WINDOW FILTERS")
    print("=" * 60)

    # Recent documents with drafts
    with_drafts = HybridFilterBuilder.build_time_window_filter(
        "2024-01-01", "2024-12-31", include_drafts=True
    )
    print("\n  Including drafts:")
    print(f"  {json.dumps(with_drafts, indent=4)}")

    # Published only
    published_only = HybridFilterBuilder.build_time_window_filter(
        "2024-01-01", "2024-12-31", include_drafts=False
    )
    print("\n  Published only:")
    print(f"  {json.dumps(published_only, indent=4)}")

    print("\n✅ Time window filters built")


def demonstrate_filter_composition():
    """Demonstrate composing multiple complex filters."""
    print("\n" + "=" * 60)
    print("4. FILTER COMPOSITION")
    print("=" * 60)

    # Compose multiple filters together
    security_filter = {
        "$and": [
            {"department": {"$in": ["Technology", "Engineering"]}},
            {"clearance_level": {"$in": ["public", "internal"]}},
        ]
    }

    quality_filter = HybridFilterBuilder.build_complex_production_filter()

    compliance_filter = HybridFilterBuilder.build_geo_compliance_filter(["US"])

    # Merge all filters
    merged = {"$and": [security_filter, quality_filter, compliance_filter]}

    print(f"\n  Security filter: {json.dumps(security_filter, indent=2)}")
    print(f"\n  Quality filter: {json.dumps(quality_filter, indent=2)}")
    print(f"\n  Compliance filter: {json.dumps(compliance_filter, indent=2)}")
    print("\n  Merged filter:")
    print(f"  {json.dumps(merged, indent=2)}")

    # Count total conditions
    def count_conditions(filter_dict: Dict) -> int:
        """Recursively count conditions in a filter."""
        count = 0
        for key, value in filter_dict.items():
            if key in ["$and", "$or"]:
                for item in value:
                    count += count_conditions(item)
            elif key == "$not":
                count += count_conditions(value)
            else:
                count += 1
        return count

    total = count_conditions(merged)
    print(f"\n  Total filter conditions: {total}")
    print("\n✅ Filters composed successfully")


def demonstrate_filter_optimization():
    """Demonstrate filter optimization techniques."""
    print("\n" + "=" * 60)
    print("5. FILTER OPTIMIZATION")
    print("=" * 60)

    def optimize_filter(filter_dict: Dict) -> Dict:
        """Optimize filter by flattening and deduplicating."""
        # Convert to string for comparison
        filter_str = json.dumps(filter_dict, sort_keys=True)

        # Simple optimizations:
        # 1. Remove redundant $and with single condition
        # 2. Merge nested $and operations
        # 3. Remove duplicate conditions

        return filter_dict  # In production, implement actual optimization

    # Example: redundant nesting
    redundant = {"$and": [{"$and": [{"department": "Technology"}]}]}

    print("\n  Redundant nesting:")
    print(f"  Input:  {json.dumps(redundant)}")
    print("\n  Optimization tips:")
    print("  - Flatten nested $and operators")
    print("  - Remove single-condition $and wrappers")
    print("  - Deduplicate identical conditions")
    print("  - Push $eq to top level when possible")

    print("\n✅ Filter optimization techniques documented")


def main():
    """Run all hybrid multi-criteria demonstrations."""
    print("=" * 60)
    print("HYBRID MULTI-CRITERIA FILTERING")
    print("=" * 60)

    demonstrate_nested_filter_building()
    demonstrate_geo_compliance()
    demonstrate_time_windows()
    demonstrate_filter_composition()
    demonstrate_filter_optimization()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Complex filtering requires careful composition")
    print("   of AND/OR logic with proper nesting.")


if __name__ == "__main__":
    main()
