"""
02_document_type_filtering.py

Document type and format-based filtering.
Demonstrates filtering by file format, document classification, and status.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List


class DocumentType(str, Enum):
    """Types of documents."""

    REPORT = "report"
    POLICY = "policy"
    MEMO = "memo"
    WHITEPAPER = "whitepaper"
    RESEARCH_PAPER = "research_paper"
    OFFICIAL_REPORT = "official_report"
    GUIDE = "guide"
    TEMPLATE = "template"
    STRATEGY = "strategy"
    REVIEW = "review"


class FileFormat(str, Enum):
    """Supported file formats."""

    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    MD = "md"
    TXT = "txt"
    HTML = "html"


class DocumentStatus(str, Enum):
    """Document lifecycle status."""

    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class Classification(str, Enum):
    """Document classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"


@dataclass
class Document:
    """Document with type and format metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentTypeFilter:
    """Filter documents by type, format, status, and classification."""

    @staticmethod
    def by_type(*types: DocumentType) -> Callable[[Document], bool]:
        """Filter by document types."""
        type_values = [t.value for t in types]
        return lambda doc: doc.metadata.get("doc_type") in type_values

    @staticmethod
    def by_format(*formats: FileFormat) -> Callable[[Document], bool]:
        """Filter by file formats."""
        format_values = [f.value for f in formats]
        return lambda doc: doc.metadata.get("file_format") in format_values

    @staticmethod
    def by_status(*statuses: DocumentStatus) -> Callable[[Document], bool]:
        """Filter by document status."""
        status_values = [s.value for s in statuses]
        return lambda doc: doc.metadata.get("status") in status_values

    @staticmethod
    def exclude_status(*statuses: DocumentStatus) -> Callable[[Document], bool]:
        """Exclude documents with certain statuses."""
        status_values = [s.value for s in statuses]
        return lambda doc: doc.metadata.get("status") not in status_values

    @staticmethod
    def by_classification(max_level: Classification) -> Callable[[Document], bool]:
        """Filter by maximum classification level (inclusive)."""
        levels = list(Classification)
        max_index = levels.index(max_level)
        allowed = [c.value for c in levels[: max_index + 1]]
        return lambda doc: doc.metadata.get("classification") in allowed

    @staticmethod
    def not_classification(*levels: Classification) -> Callable[[Document], bool]:
        """Exclude certain classification levels."""
        level_values = [l.value for l in levels]
        return lambda doc: doc.metadata.get("classification") not in level_values

    @staticmethod
    def combine_and(*filters: Callable) -> Callable[[Document], bool]:
        """Combine filters with AND logic."""

        def combined(doc: Document) -> bool:
            return all(f(doc) for f in filters if f)

        return combined

    @staticmethod
    def apply_filters(
        documents: List[Document], *filters: Callable[[Document], bool]
    ) -> List[Document]:
        """Apply multiple filters to documents."""
        combined = DocumentTypeFilter.combine_and(*filters)
        return [doc for doc in documents if combined(doc)]


def create_sample_documents() -> List[Document]:
    """Create sample documents with type/format metadata."""
    return [
        Document(
            "Annual financial report with detailed analysis",
            {
                "doc_type": "report",
                "file_format": "pdf",
                "status": "published",
                "classification": "confidential",
                "department": "Finance",
                "year": 2024,
            },
        ),
        Document(
            "Remote work policy guidelines",
            {
                "doc_type": "policy",
                "file_format": "docx",
                "status": "approved",
                "classification": "internal",
                "department": "HR",
                "year": 2024,
            },
        ),
        Document(
            "Project X draft proposal",
            {
                "doc_type": "memo",
                "file_format": "docx",
                "status": "draft",
                "classification": "internal",
                "department": "Technology",
                "year": 2024,
            },
        ),
        Document(
            "AI research whitepaper on transformers",
            {
                "doc_type": "whitepaper",
                "file_format": "pdf",
                "status": "published",
                "classification": "public",
                "department": "Technology",
                "year": 2023,
            },
        ),
        Document(
            "Strategic plan template for departments",
            {
                "doc_type": "template",
                "file_format": "xlsx",
                "status": "published",
                "classification": "internal",
                "department": "Operations",
                "year": 2023,
            },
        ),
        Document(
            "GDPR compliance review document",
            {
                "doc_type": "review",
                "file_format": "pdf",
                "status": "approved",
                "classification": "restricted",
                "department": "Legal",
                "year": 2024,
            },
        ),
        Document(
            "Archived server migration guide",
            {
                "doc_type": "guide",
                "file_format": "md",
                "status": "archived",
                "classification": "internal",
                "department": "Technology",
                "year": 2022,
            },
        ),
        Document(
            "Secret product roadmap",
            {
                "doc_type": "strategy",
                "file_format": "pptx",
                "status": "draft",
                "classification": "secret",
                "department": "Product",
                "year": 2025,
            },
        ),
    ]


def demonstrate_type_filtering():
    """Demonstrate document type filtering."""
    print("\n" + "=" * 60)
    print("1. RESEARCH CONTENT: Whitepapers and research papers")
    print("=" * 60)

    docs = create_sample_documents()
    dtf = DocumentTypeFilter

    filtered = dtf.apply_filters(
        docs, dtf.by_type(DocumentType.WHITEPAPER, DocumentType.RESEARCH_PAPER)
    )

    for doc in filtered:
        print(f"  [{doc.metadata['doc_type']}] {doc.content[:60]}...")

    print(f"\n✅ {len(filtered)}/{len(docs)} documents are research content")


def demonstrate_format_filtering():
    """Demonstrate file format filtering."""
    print("\n" + "=" * 60)
    print("2. EDITABLE FORMATS: DOCX and XLSX only")
    print("=" * 60)

    docs = create_sample_documents()
    dtf = DocumentTypeFilter

    filtered = dtf.apply_filters(docs, dtf.by_format(FileFormat.DOCX, FileFormat.XLSX))

    for doc in filtered:
        print(
            f"  [{doc.metadata['file_format']}] {doc.metadata['doc_type']}: {doc.content[:50]}..."
        )

    print(f"\n✅ {len(filtered)}/{len(docs)} documents in editable formats")


def demonstrate_status_filtering():
    """Demonstrate document status filtering."""
    print("\n" + "=" * 60)
    print("3. READY FOR USE: Published or approved only")
    print("=" * 60)

    docs = create_sample_documents()
    dtf = DocumentTypeFilter

    filtered = dtf.apply_filters(
        docs, dtf.by_status(DocumentStatus.PUBLISHED, DocumentStatus.APPROVED)
    )

    for doc in filtered:
        print(
            f"  [{doc.metadata['status']}] {doc.metadata['doc_type']}: {doc.content[:50]}..."
        )

    print(f"\n✅ {len(filtered)}/{len(docs)} documents are ready for use")


def demonstrate_exclusion_filtering():
    """Demonstrate excluding certain document states."""
    print("\n" + "=" * 60)
    print("4. EXCLUDE: No drafts or archived")
    print("=" * 60)

    docs = create_sample_documents()
    dtf = DocumentTypeFilter

    filtered = dtf.apply_filters(
        docs, dtf.exclude_status(DocumentStatus.DRAFT, DocumentStatus.ARCHIVED)
    )

    for doc in filtered:
        print(
            f"  [{doc.metadata['status']}] {doc.metadata['doc_type']}: {doc.content[:50]}..."
        )

    excluded = len(docs) - len(filtered)
    print(f"\n✅ {len(filtered)}/{len(docs)} documents kept, {excluded} excluded")


def demonstrate_classification_filtering():
    """Demonstrate classification-based access filtering."""
    print("\n" + "=" * 60)
    print("5. ACCESS LEVEL: Internal or below (no confidential/restricted/secret)")
    print("=" * 60)

    docs = create_sample_documents()
    dtf = DocumentTypeFilter

    filtered = dtf.apply_filters(
        docs,
        dtf.by_classification(Classification.INTERNAL),
        dtf.exclude_status(DocumentStatus.ARCHIVED),
    )

    for doc in filtered:
        print(
            f"  [{doc.metadata['classification']}] {doc.metadata['doc_type']}: "
            f"{doc.content[:50]}..."
        )

    print(f"\n✅ {len(filtered)}/{len(docs)} documents accessible at this level")


def demonstrate_combined_type_filtering():
    """Demonstrate comprehensive type-based filtering."""
    print("\n" + "=" * 60)
    print("6. COMPREHENSIVE: Published PDFs + reports/policies + internal or lower")
    print("=" * 60)

    docs = create_sample_documents()
    dtf = DocumentTypeFilter

    filtered = dtf.apply_filters(
        docs,
        dtf.by_type(DocumentType.REPORT, DocumentType.POLICY, DocumentType.WHITEPAPER),
        dtf.by_format(FileFormat.PDF),
        dtf.by_status(DocumentStatus.PUBLISHED),
        dtf.by_classification(Classification.INTERNAL),
    )

    for doc in filtered:
        print(
            f"  [{doc.metadata['doc_type']}] [{doc.metadata['file_format']}] "
            f"[{doc.metadata['classification']}] {doc.content[:40]}..."
        )

    print(f"\n✅ {len(filtered)}/{len(docs)} documents match all criteria")


def demonstrate_vector_db_filter_building():
    """Demonstrate building vector DB filters from document type criteria."""
    print("\n" + "=" * 60)
    print("7. VECTOR DB FILTER: ChromaDB-compatible filter")
    print("=" * 60)

    # This filter would be used with a vector database
    chroma_filter = {
        "$and": [
            {"file_type": {"$in": ["pdf", "docx"]}},
            {
                "document_type": {
                    "$in": ["whitepaper", "research_paper", "official_report"]
                }
            },
            {"status": {"$eq": "approved"}},
            {"classification": {"$ne": "confidential"}},
            {"is_draft": {"$eq": False}},
        ]
    }

    print("\n  Filter configuration:")
    import json

    print(f"  {json.dumps(chroma_filter, indent=2)}")
    print("\n  This filter:")
    print("  - Only PDF and DOCX formats")
    print("  - Only research content types")
    print("  - Must be approved")
    print("  - Excludes confidential documents")
    print("  - No drafts")

    print("\n✅ Vector database filter ready for use")


def main():
    """Run all document type filtering demonstrations."""
    print("=" * 60)
    print("DOCUMENT TYPE & FORMAT FILTERING DEMONSTRATION")
    print("=" * 60)

    demonstrate_type_filtering()
    demonstrate_format_filtering()
    demonstrate_status_filtering()
    demonstrate_exclusion_filtering()
    demonstrate_classification_filtering()
    demonstrate_combined_type_filtering()
    demonstrate_vector_db_filter_building()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Document type filtering ensures users only access")
    print("   content in appropriate formats and classifications.")


if __name__ == "__main__":
    main()
