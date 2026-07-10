"""
01_basic_chromadb.py

Basic ChromaDB filtering examples.
Covers inclusion, exclusion, multiple inclusions, and complex multi-field filtering.
"""

import os

from jet.adapters.langchain.factory import get_openai_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def create_sample_vectorstore():
    """Create a sample vector store with metadata for demonstration."""
    embeddings = get_openai_embeddings()

    documents = [
        Document(
            page_content="Our AI initiatives focus on natural language processing and computer vision.",
            metadata={
                "department": "Technology",
                "topic": "AI",
                "status": "active",
                "year": 2024,
            },
        ),
        Document(
            page_content="Machine learning models have improved customer segmentation by 40%.",
            metadata={
                "department": "Technology",
                "topic": "Machine Learning",
                "status": "active",
                "year": 2024,
            },
        ),
        Document(
            page_content="Employee onboarding process has been streamlined with new HR software.",
            metadata={
                "department": "HR",
                "topic": "Onboarding",
                "status": "active",
                "year": 2023,
            },
        ),
        Document(
            page_content="Company policies regarding remote work have been updated for 2024.",
            metadata={
                "department": "HR",
                "topic": "Policies",
                "status": "active",
                "year": 2024,
            },
        ),
        Document(
            page_content="Budget approval process requires department head sign-off for amounts over $10,000.",
            metadata={
                "department": "Finance",
                "topic": "Budget",
                "status": "active",
                "year": 2023,
            },
        ),
        Document(
            page_content="Q4 financial projections show 15% growth in recurring revenue.",
            metadata={
                "department": "Finance",
                "topic": "Revenue",
                "status": "draft",
                "year": 2024,
            },
        ),
    ]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="demo_domain_filtering",
    )

    return vectorstore


def demonstrate_inclusion_filter(vectorstore):
    """Single inclusion filter - search only in Technology department."""
    print("\n" + "=" * 60)
    print("1. SINGLE INCLUSION: Technology department only")
    print("=" * 60)

    docs = vectorstore.similarity_search(
        query="What are our AI initiatives?", k=5, filter={"department": "Technology"}
    )

    for i, doc in enumerate(docs, 1):
        print(f"\nResult {i}:")
        print(f"  Content: {doc.page_content}")
        print(f"  Department: {doc.metadata.get('department')}")
        print(f"  Topic: {doc.metadata.get('topic')}")

    print(f"\n✅ Retrieved {len(docs)} documents from Technology department")


def demonstrate_exclusion_filter(vectorstore):
    """Exclusion filter - exclude HR documents."""
    print("\n" + "=" * 60)
    print("2. EXCLUSION: Exclude HR department")
    print("=" * 60)

    docs = vectorstore.similarity_search(
        query="Company policies", k=5, filter={"department": {"$ne": "HR"}}
    )

    for i, doc in enumerate(docs, 1):
        print(f"\nResult {i}:")
        print(f"  Content: {doc.page_content}")
        print(f"  Department: {doc.metadata.get('department')}")

    # Verify no HR documents
    hr_docs = [d for d in docs if d.metadata.get("department") == "HR"]
    print(f"\n✅ Retrieved {len(docs)} documents, {len(hr_docs)} from HR (should be 0)")


def demonstrate_multiple_inclusions(vectorstore):
    """Multiple inclusions filter - include multiple departments."""
    print("\n" + "=" * 60)
    print("3. MULTIPLE INCLUSIONS: Finance and Operations")
    print("=" * 60)

    docs = vectorstore.similarity_search(
        query="Budget approvals",
        k=5,
        filter={"department": {"$in": ["Finance", "Operations"]}},
    )

    for i, doc in enumerate(docs, 1):
        print(f"\nResult {i}:")
        print(f"  Content: {doc.page_content}")
        print(f"  Department: {doc.metadata.get('department')}")

    departments = set(d.metadata.get("department") for d in docs)
    print(f"\n✅ Retrieved {len(docs)} documents from departments: {departments}")


def demonstrate_complex_filter(vectorstore):
    """Complex multi-field filtering with AND logic."""
    print("\n" + "=" * 60)
    print("4. COMPLEX MULTI-FIELD: Technology + AI/ML + active status")
    print("=" * 60)

    docs = vectorstore.similarity_search(
        query="Recent AI initiatives",
        k=5,
        filter={
            "$and": [
                {"department": "Technology"},
                {"topic": {"$in": ["AI", "Machine Learning"]}},
                {"status": "active"},
            ]
        },
    )

    for i, doc in enumerate(docs, 1):
        print(f"\nResult {i}:")
        print(f"  Content: {doc.page_content}")
        print(f"  Department: {doc.metadata.get('department')}")
        print(f"  Topic: {doc.metadata.get('topic')}")
        print(f"  Status: {doc.metadata.get('status')}")

    print(f"\n✅ Retrieved {len(docs)} documents matching all criteria")


def main():
    """Run all ChromaDB filtering demonstrations."""
    print("=" * 60)
    print("CHROMADB DOMAIN FILTERING DEMONSTRATION")
    print("=" * 60)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set. Using mock mode.")
        print("Set your API key: export OPENAI_API_KEY='your-key-here'\n")
        return

    vectorstore = create_sample_vectorstore()

    demonstrate_inclusion_filter(vectorstore)
    demonstrate_exclusion_filter(vectorstore)
    demonstrate_multiple_inclusions(vectorstore)
    demonstrate_complex_filter(vectorstore)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
