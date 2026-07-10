"""
02_advanced_chromadb.py

Advanced ChromaDB filtering with dynamic filter building and metadata management.
Extends basic patterns with reusable filter builders and query analysis.
"""

import os
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class ChromaFilterBuilder:
    """Reusable filter builder for ChromaDB queries."""

    @staticmethod
    def include(field: str, value: Any) -> Dict:
        """Include exact value."""
        return {field: value}

    @staticmethod
    def include_any(field: str, values: List[Any]) -> Dict:
        """Include any of the values."""
        if len(values) == 1:
            return {field: values[0]}
        return {field: {"$in": values}}

    @staticmethod
    def exclude(field: str, value: Any) -> Dict:
        """Exclude specific value."""
        return {field: {"$ne": value}}

    @staticmethod
    def exclude_any(field: str, values: List[Any]) -> Dict:
        """Exclude any of the values."""
        return {field: {"$nin": values}}

    @staticmethod
    def greater_equal(field: str, value: Any) -> Dict:
        """Greater than or equal."""
        return {field: {"$gte": value}}

    @staticmethod
    def less_equal(field: str, value: Any) -> Dict:
        """Less than or equal."""
        return {field: {"$lte": value}}

    @staticmethod
    def range_filter(field: str, min_val: Any = None, max_val: Any = None) -> Dict:
        """Range filter with optional min/max."""
        conditions = {}
        if min_val is not None:
            conditions["$gte"] = min_val
        if max_val is not None:
            conditions["$lte"] = max_val
        return {field: conditions} if conditions else {}

    @staticmethod
    def combine_and(*filters: Dict) -> Dict:
        """Combine filters with AND logic."""
        valid_filters = [f for f in filters if f]
        if not valid_filters:
            return {}
        if len(valid_filters) == 1:
            return valid_filters[0]
        return {"$and": valid_filters}

    @staticmethod
    def combine_or(*filters: Dict) -> Dict:
        """Combine filters with OR logic."""
        valid_filters = [f for f in filters if f]
        if not valid_filters:
            return {}
        if len(valid_filters) == 1:
            return valid_filters[0]
        return {"$or": valid_filters}


def create_advanced_vectorstore():
    """Create a vector store with rich metadata."""
    embeddings = OpenAIEmbeddings()

    documents = [
        Document(
            page_content="Cloud migration strategy using AWS and Azure hybrid approach.",
            metadata={
                "department": "Technology",
                "sub_department": "Infrastructure",
                "topic": "Cloud",
                "year": 2024,
                "quarter": "Q2",
                "author": "Alice Johnson",
                "credibility_score": 0.92,
                "status": "published",
                "tags": ["cloud", "AWS", "migration"],
            },
        ),
        Document(
            page_content="AI ethics guidelines for machine learning model deployment.",
            metadata={
                "department": "Technology",
                "sub_department": "AI Research",
                "topic": "AI Ethics",
                "year": 2024,
                "quarter": "Q1",
                "author": "Bob Smith",
                "credibility_score": 0.88,
                "status": "published",
                "tags": ["AI", "ethics", "ML"],
            },
        ),
        Document(
            page_content="Employee wellness program results showing 25% improvement.",
            metadata={
                "department": "HR",
                "sub_department": "Benefits",
                "topic": "Wellness",
                "year": 2023,
                "quarter": "Q4",
                "author": "Carol Davis",
                "credibility_score": 0.75,
                "status": "published",
                "tags": ["wellness", "benefits", "employees"],
            },
        ),
        Document(
            page_content="Remote work policy update for hybrid workforce management.",
            metadata={
                "department": "HR",
                "sub_department": "Policy",
                "topic": "Remote Work",
                "year": 2024,
                "quarter": "Q2",
                "author": "Diana Wilson",
                "credibility_score": 0.70,
                "status": "draft",
                "tags": ["remote", "policy", "hybrid"],
            },
        ),
        Document(
            page_content="Q3 financial analysis shows 15% revenue growth in SaaS products.",
            metadata={
                "department": "Finance",
                "sub_department": "Analysis",
                "topic": "Revenue",
                "year": 2023,
                "quarter": "Q3",
                "author": "Eve Brown",
                "credibility_score": 0.95,
                "status": "published",
                "tags": ["revenue", "SaaS", "growth"],
            },
        ),
        Document(
            page_content="Budget planning framework for 2025 fiscal year projections.",
            metadata={
                "department": "Finance",
                "sub_department": "Planning",
                "topic": "Budget",
                "year": 2024,
                "quarter": "Q3",
                "author": "Frank Miller",
                "credibility_score": 0.65,
                "status": "draft",
                "tags": ["budget", "planning", "2025"],
            },
        ),
        Document(
            page_content="Patent application for novel machine learning architecture.",
            metadata={
                "department": "Legal",
                "sub_department": "IP",
                "topic": "Patent",
                "year": 2024,
                "quarter": "Q1",
                "author": "Grace Lee",
                "credibility_score": 0.98,
                "status": "confidential",
                "tags": ["patent", "ML", "architecture"],
            },
        ),
        Document(
            page_content="GDPR compliance audit results for European operations.",
            metadata={
                "department": "Legal",
                "sub_department": "Compliance",
                "topic": "GDPR",
                "year": 2023,
                "quarter": "Q4",
                "author": "Henry Taylor",
                "credibility_score": 0.90,
                "status": "published",
                "tags": ["GDPR", "compliance", "audit"],
            },
        ),
    ]

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="demo_advanced_filtering",
    )


def demonstrate_dynamic_filtering(vectorstore):
    """Demonstrate dynamic filter building with the builder class."""
    print("\n" + "=" * 60)
    print("1. DYNAMIC FILTER: Tech dept + published + high credibility")
    print("=" * 60)

    fb = ChromaFilterBuilder
    filter_dict = fb.combine_and(
        fb.include("department", "Technology"),
        fb.include("status", "published"),
        fb.greater_equal("credibility_score", 0.85),
    )

    docs = vectorstore.similarity_search(
        query="Cloud and AI initiatives", k=5, filter=filter_dict
    )

    for i, doc in enumerate(docs, 1):
        print(f"\n  Result {i}:")
        print(f"    Content: {doc.page_content[:80]}...")
        print(f"    Dept: {doc.metadata['department']}")
        print(f"    Score: {doc.metadata['credibility_score']}")

    print(f"\n  ✅ Retrieved {len(docs)} documents")


def demonstrate_or_logic(vectorstore):
    """Demonstrate OR logic across departments."""
    print("\n" + "=" * 60)
    print("2. OR LOGIC: Technology OR Legal, published only")
    print("=" * 60)

    fb = ChromaFilterBuilder
    filter_dict = fb.combine_and(
        fb.combine_or(
            fb.include("department", "Technology"), fb.include("department", "Legal")
        ),
        fb.include("status", "published"),
    )

    docs = vectorstore.similarity_search(
        query="Compliance and technology", k=5, filter=filter_dict
    )

    for i, doc in enumerate(docs, 1):
        print(f"\n  Result {i}:")
        print(f"    Content: {doc.page_content[:80]}...")
        print(f"    Dept: {doc.metadata['department']}")
        print(f"    Status: {doc.metadata['status']}")

    print(f"\n  ✅ Retrieved {len(docs)} documents")


def demonstrate_range_with_exclusion(vectorstore):
    """Demonstrate range filters with exclusions."""
    print("\n" + "=" * 60)
    print("3. RANGE + EXCLUSION: 2024, not drafts, not HR")
    print("=" * 60)

    fb = ChromaFilterBuilder
    filter_dict = fb.combine_and(
        fb.include("year", 2024),
        fb.exclude("status", "draft"),
        fb.exclude("department", "HR"),
    )

    docs = vectorstore.similarity_search(
        query="Strategic initiatives", k=5, filter=filter_dict
    )

    for i, doc in enumerate(docs, 1):
        print(f"\n  Result {i}:")
        print(f"    Content: {doc.page_content[:80]}...")
        print(f"    Dept: {doc.metadata['department']}")
        print(f"    Year: {doc.metadata['year']}")
        print(f"    Status: {doc.metadata['status']}")

    print(f"\n  ✅ Retrieved {len(docs)} documents")


def demonstrate_tag_filtering(vectorstore):
    """Demonstrate filtering by list-type metadata (tags)."""
    print("\n" + "=" * 60)
    print("4. TAG FILTERING: Documents tagged with 'ML' or 'AI'")
    print("=" * 60)

    fb = ChromaFilterBuilder
    filter_dict = fb.include_any("tags", ["AI", "ML", "machine learning"])

    docs = vectorstore.similarity_search(
        query="Machine learning projects", k=5, filter=filter_dict
    )

    for i, doc in enumerate(docs, 1):
        print(f"\n  Result {i}:")
        print(f"    Content: {doc.page_content[:80]}...")
        print(f"    Tags: {doc.metadata['tags']}")

    print(f"\n  ✅ Retrieved {len(docs)} documents")


def demonstrate_multi_level_filtering(vectorstore):
    """Demonstrate filtering across multiple metadata levels."""
    print("\n" + "=" * 60)
    print("5. MULTI-LEVEL: Published + (Q1 or Q2) + score >= 0.8")
    print("=" * 60)

    fb = ChromaFilterBuilder
    filter_dict = fb.combine_and(
        fb.include("status", "published"),
        fb.combine_or(fb.include("quarter", "Q1"), fb.include("quarter", "Q2")),
        fb.greater_equal("credibility_score", 0.8),
    )

    docs = vectorstore.similarity_search(
        query="First half achievements", k=5, filter=filter_dict
    )

    for i, doc in enumerate(docs, 1):
        print(f"\n  Result {i}:")
        print(f"    Content: {doc.page_content[:80]}...")
        print(f"    Quarter: {doc.metadata['quarter']}")
        print(f"    Score: {doc.metadata['credibility_score']}")
        print(f"    Status: {doc.metadata['status']}")

    print(f"\n  ✅ Retrieved {len(docs)} documents")


def main():
    """Run all advanced ChromaDB demonstrations."""
    print("=" * 60)
    print("ADVANCED CHROMADB FILTERING DEMONSTRATION")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set.")
        print("Set your API key: export OPENAI_API_KEY='your-key-here'\n")
        return

    vectorstore = create_advanced_vectorstore()

    demonstrate_dynamic_filtering(vectorstore)
    demonstrate_or_logic(vectorstore)
    demonstrate_range_with_exclusion(vectorstore)
    demonstrate_tag_filtering(vectorstore)
    demonstrate_multi_level_filtering(vectorstore)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 The ChromaFilterBuilder provides reusable, composable filter patterns.")


if __name__ == "__main__":
    main()
