"""
03_basic_pydantic_extraction.py

Basic structured filter extraction using Pydantic models.
Demonstrates foundational patterns for LLM-powered query understanding
without requiring actual LLM calls.

This is a NEW basic example not in the original markdown.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Department(str, Enum):
    """Valid departments."""

    TECHNOLOGY = "Technology"
    HR = "HR"
    FINANCE = "Finance"
    LEGAL = "Legal"
    MARKETING = "Marketing"


class DocumentType(str, Enum):
    """Valid document types."""

    REPORT = "report"
    POLICY = "policy"
    GUIDE = "guide"
    RESEARCH_PAPER = "research_paper"
    MEMO = "memo"
    STRATEGY = "strategy"


class FilterCriteria(BaseModel):
    """
    Schema for extracted filter criteria from user queries.
    This demonstrates the structure that an LLM would populate.
    """

    departments: Optional[List[Department]] = Field(
        default=None, description="List of departments to include in search"
    )
    exclude_departments: Optional[List[Department]] = Field(
        default=None, description="List of departments to exclude from search"
    )
    min_year: Optional[int] = Field(
        default=None,
        ge=2000,
        le=datetime.now().year,
        description="Minimum publication year",
    )
    max_year: Optional[int] = Field(
        default=None,
        ge=2000,
        le=datetime.now().year,
        description="Maximum publication year",
    )
    doc_types: Optional[List[DocumentType]] = Field(
        default=None, description="Document types to include"
    )
    min_credibility: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum credibility score (0-1)"
    )
    exclude_drafts: bool = Field(
        default=False, description="Whether to exclude draft documents"
    )
    keywords: Optional[List[str]] = Field(
        default=None, description="Keywords to search for in content"
    )

    @field_validator("max_year")
    @classmethod
    def validate_year_range(cls, v, info):
        """Validate that min_year <= max_year."""
        if v is not None and "min_year" in info.data:
            min_year = info.data["min_year"]
            if min_year is not None and v < min_year:
                raise ValueError(f"max_year ({v}) must be >= min_year ({min_year})")
        return v


class FilterBuilder:
    """
    Convert FilterCriteria into actual filter implementations.
    This mimics what vector databases do internally.
    """

    @staticmethod
    def build_chromadb_filter(criteria: FilterCriteria) -> Dict[str, Any]:
        """Build ChromaDB-compatible filter."""
        conditions = []

        if criteria.departments:
            departments = [d.value for d in criteria.departments]
            if len(departments) == 1:
                conditions.append({"department": departments[0]})
            else:
                conditions.append({"department": {"$in": departments}})

        if criteria.exclude_departments:
            for dept in criteria.exclude_departments:
                conditions.append({"department": {"$ne": dept.value}})

        if criteria.min_year is not None:
            conditions.append({"year": {"$gte": criteria.min_year}})

        if criteria.max_year is not None:
            conditions.append({"year": {"$lte": criteria.max_year}})

        if criteria.doc_types:
            types = [t.value for t in criteria.doc_types]
            if len(types) == 1:
                conditions.append({"doc_type": types[0]})
            else:
                conditions.append({"doc_type": {"$in": types}})

        if criteria.min_credibility is not None:
            conditions.append({"credibility_score": {"$gte": criteria.min_credibility}})

        if criteria.exclude_drafts:
            conditions.append({"status": {"$ne": "draft"}})

        if len(conditions) == 0:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    @staticmethod
    def build_generic_filter(criteria: FilterCriteria) -> List[callable]:
        """Build generic Python filter functions."""
        filters = []

        if criteria.departments:
            departments = [d.value for d in criteria.departments]
            filters.append(lambda meta: meta.get("department") in departments)

        if criteria.exclude_departments:
            exclude = [d.value for d in criteria.exclude_departments]
            filters.append(lambda meta: meta.get("department") not in exclude)

        if criteria.min_year is not None:
            min_y = criteria.min_year
            filters.append(lambda meta: meta.get("year", 0) >= min_y)

        if criteria.max_year is not None:
            max_y = criteria.max_year
            filters.append(lambda meta: meta.get("year", float("inf")) <= max_y)

        if criteria.doc_types:
            types = [t.value for t in criteria.doc_types]
            filters.append(lambda meta: meta.get("doc_type") in types)

        if criteria.min_credibility is not None:
            min_c = criteria.min_credibility
            filters.append(lambda meta: meta.get("credibility_score", 0) >= min_c)

        if criteria.exclude_drafts:
            filters.append(lambda meta: meta.get("status") != "draft")

        return filters


def demonstrate_simple_extraction():
    """Demonstrate simple filter extraction."""
    print("\n" + "=" * 60)
    print("1. SIMPLE DEPARTMENT FILTER")
    print("=" * 60)

    # Simulate what an LLM would extract from:
    # "Show me technology documents from 2024"
    criteria = FilterCriteria(
        departments=[Department.TECHNOLOGY], min_year=2024, max_year=2024
    )

    chroma_filter = FilterBuilder.build_chromadb_filter(criteria)
    print("\nInput: 'Show me technology documents from 2024'")
    print(f"Extracted criteria: {criteria.model_dump()}")
    print(f"ChromaDB filter: {chroma_filter}")
    print("\n✅ Filter built successfully")


def demonstrate_complex_extraction():
    """Demonstrate complex filter extraction with exclusions."""
    print("\n" + "=" * 60)
    print("2. COMPLEX FILTER WITH EXCLUSIONS")
    print("=" * 60)

    # Simulate what an LLM would extract from:
    # "Find published reports and strategies from Technology or Finance,
    #  year 2023-2024, exclude drafts, minimum credibility 0.8"
    criteria = FilterCriteria(
        departments=[Department.TECHNOLOGY, Department.FINANCE],
        min_year=2023,
        max_year=2024,
        doc_types=[DocumentType.REPORT, DocumentType.STRATEGY],
        min_credibility=0.8,
        exclude_drafts=True,
    )

    chroma_filter = FilterBuilder.build_chromadb_filter(criteria)
    print(
        "\nInput: 'Find published reports and strategies from Technology or Finance, 2023-2024, exclude drafts, min credibility 0.8'"
    )
    print(f"Extracted criteria: {criteria.model_dump()}")
    print("ChromaDB filter:")
    import json

    print(json.dumps(chroma_filter, indent=2))
    print("\n✅ Complex filter built successfully")


def demonstrate_validation():
    """Demonstrate Pydantic validation."""
    print("\n" + "=" * 60)
    print("3. INPUT VALIDATION")
    print("=" * 60)

    # Test valid input
    try:
        criteria = FilterCriteria(
            departments=[Department.TECHNOLOGY], min_year=2023, max_year=2024
        )
        print(f"\n✅ Valid criteria accepted: {criteria.model_dump()}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

    # Test invalid year range
    try:
        criteria = FilterCriteria(
            min_year=2024,
            max_year=2023,  # Should fail validation
        )
        print("❌ Should have raised validation error")
    except Exception as e:
        print(f"\n✅ Correctly rejected invalid year range: {e}")

    # Test invalid credibility score
    try:
        criteria = FilterCriteria(
            min_credibility=1.5  # Should fail, > 1.0
        )
        print("❌ Should have raised validation error")
    except Exception as e:
        print(f"\n✅ Correctly rejected invalid credibility score: {e}")


def demonstrate_empty_filter():
    """Demonstrate handling of empty/unspecified filters."""
    print("\n" + "=" * 60)
    print("4. EMPTY FILTER HANDLING")
    print("=" * 60)

    # Simulate a query with no specific filters
    criteria = FilterCriteria()

    chroma_filter = FilterBuilder.build_chromadb_filter(criteria)
    generic_filters = FilterBuilder.build_generic_filter(criteria)

    print("\nInput: 'Tell me about our company' (no specific filters)")
    print(f"Extracted criteria: {criteria.model_dump()}")
    print(f"ChromaDB filter: {chroma_filter}")
    print(f"Generic filters count: {len(generic_filters)}")
    print("\n✅ Empty filter handled correctly (no restrictions applied)")


def demonstrate_filter_composition():
    """Demonstrate composing multiple filters together."""
    print("\n" + "=" * 60)
    print("5. FILTER COMPOSITION WITH SECURITY OVERLAY")
    print("=" * 60)

    # User's query filter
    user_criteria = FilterCriteria(departments=[Department.TECHNOLOGY], min_year=2024)

    # Security overlay (would come from user permissions)
    security_criteria = FilterCriteria(
        departments=[
            Department.TECHNOLOGY,
            Department.FINANCE,
        ],  # User can access these
        exclude_drafts=True,  # No drafts
        min_credibility=0.5,  # Minimum quality
    )

    user_filter = FilterBuilder.build_chromadb_filter(user_criteria)
    security_filter = FilterBuilder.build_chromadb_filter(security_criteria)

    # Merge filters
    merged_filter = (
        {"$and": [user_filter, security_filter]} if user_filter else security_filter
    )

    print(f"\nUser query filter: {user_filter}")
    print(f"Security overlay: {security_filter}")
    print(f"Merged filter: {merged_filter}")
    print("\n✅ Filters composed successfully")


def main():
    """Run all basic Pydantic extraction demonstrations."""
    print("=" * 60)
    print("BASIC PYDANTIC FILTER EXTRACTION")
    print("=" * 60)
    print("\nThis demonstrates structured filter extraction patterns")
    print("that LLMs use for natural language query understanding.\n")

    demonstrate_simple_extraction()
    demonstrate_complex_extraction()
    demonstrate_validation()
    demonstrate_empty_filter()
    demonstrate_filter_composition()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Key concepts demonstrated:")
    print("  - Structured filter criteria using Pydantic models")
    print("  - Conversion to vector database filter format")
    print("  - Input validation and error handling")
    print("  - Filter composition and security overlays")


if __name__ == "__main__":
    main()
