"""
01_credibility_scoring.py

Source credibility filtering with multi-criteria quality scoring.
Demonstrates filtering by source reputation, citation count, and credibility scores.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List


class SourceType(str, Enum):
    """Types of sources with inherent credibility levels."""

    PEER_REVIEWED = "peer_reviewed"
    OFFICIAL = "official"
    ACADEMIC = "academic"
    NEWS = "news"
    BLOG = "blog"
    OPINION = "opinion"
    SOCIAL_MEDIA = "social_media"


class CredibilityTier(str, Enum):
    """Credibility tiers for sources."""

    HIGH = "high"  # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"  # 0.3 - 0.5
    UNRELIABLE = "unreliable"  # < 0.3


@dataclass
class SourceProfile:
    """Profile defining credibility characteristics of a source."""

    name: str
    source_type: SourceType
    base_credibility: float
    requires_verification: bool = False


@dataclass
class Document:
    """Document with metadata for credibility filtering."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CredibilityScorer:
    """
    Calculate and filter by document credibility.
    Combines multiple signals into a credibility assessment.
    """

    # Base credibility scores by source type
    SOURCE_TYPE_SCORES = {
        SourceType.PEER_REVIEWED: 0.95,
        SourceType.OFFICIAL: 0.85,
        SourceType.ACADEMIC: 0.80,
        SourceType.NEWS: 0.60,
        SourceType.BLOG: 0.40,
        SourceType.OPINION: 0.30,
        SourceType.SOCIAL_MEDIA: 0.15,
    }

    # Source reputation database (simulated)
    SOURCE_REPUTATIONS = {
        "Nature": SourceProfile("Nature", SourceType.PEER_REVIEWED, 0.98),
        "Science": SourceProfile("Science", SourceType.PEER_REVIEWED, 0.97),
        "IEEE": SourceProfile("IEEE", SourceType.ACADEMIC, 0.90),
        "Reuters": SourceProfile("Reuters", SourceType.NEWS, 0.80),
        "Medium": SourceProfile("Medium", SourceType.BLOG, 0.40, True),
        "Personal Blog": SourceProfile("Personal Blog", SourceType.BLOG, 0.25, True),
    }

    @classmethod
    def get_source_credibility(cls, source_name: str) -> float:
        """Get credibility score for a known source."""
        if source_name in cls.SOURCE_REPUTATIONS:
            return cls.SOURCE_REPUTATIONS[source_name].base_credibility
        return 0.5  # Unknown source default

    @classmethod
    def calculate_document_credibility(
        cls,
        source_name: str,
        citation_count: int = 0,
        recency_days: int = 365,
        has_verification: bool = False,
    ) -> float:
        """
        Calculate overall document credibility.

        Factors:
        - Source reputation (40% weight)
        - Citation count (25% weight)
        - Recency (15% weight)
        - Verification status (20% weight)
        """
        source_score = cls.get_source_credibility(source_name)

        # Citation score (logarithmic scale)
        if citation_count == 0:
            citation_score = 0.3
        elif citation_count < 5:
            citation_score = 0.5
        elif citation_count < 20:
            citation_score = 0.7
        elif citation_count < 100:
            citation_score = 0.85
        else:
            citation_score = 0.95

        # Recency score
        if recency_days <= 30:
            recency_score = 1.0
        elif recency_days <= 90:
            recency_score = 0.9
        elif recency_days <= 365:
            recency_score = 0.7
        elif recency_days <= 730:
            recency_score = 0.5
        else:
            recency_score = 0.3

        # Verification bonus
        verification_score = 1.0 if has_verification else 0.5

        # Weighted combination
        final_score = (
            source_score * 0.40
            + citation_score * 0.25
            + recency_score * 0.15
            + verification_score * 0.20
        )

        return round(final_score, 2)

    @classmethod
    def get_credibility_tier(cls, score: float) -> CredibilityTier:
        """Map score to credibility tier."""
        if score >= 0.8:
            return CredibilityTier.HIGH
        elif score >= 0.5:
            return CredibilityTier.MEDIUM
        elif score >= 0.3:
            return CredibilityTier.LOW
        else:
            return CredibilityTier.UNRELIABLE


class CredibilityFilter:
    """Filter documents based on credibility criteria."""

    @staticmethod
    def min_score(score: float) -> Callable[[Document], bool]:
        """Filter by minimum credibility score."""
        return lambda doc: doc.metadata.get("credibility_score", 0) >= score

    @staticmethod
    def allowed_sources(sources: List[str]) -> Callable[[Document], bool]:
        """Filter by allowed sources."""
        return lambda doc: doc.metadata.get("source") in sources

    @staticmethod
    def excluded_sources(sources: List[str]) -> Callable[[Document], bool]:
        """Filter by excluded sources."""
        return lambda doc: doc.metadata.get("source") not in sources

    @staticmethod
    def min_citations(count: int) -> Callable[[Document], bool]:
        """Filter by minimum citation count."""
        return lambda doc: doc.metadata.get("citation_count", 0) >= count

    @staticmethod
    def verified_only() -> Callable[[Document], bool]:
        """Filter for verified sources only."""
        return lambda doc: doc.metadata.get("verified", False)

    @staticmethod
    def source_types(types: List[SourceType]) -> Callable[[Document], bool]:
        """Filter by source types."""
        return lambda doc: doc.metadata.get("source_type") in [t.value for t in types]

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
        combined = CredibilityFilter.combine_and(*filters)
        return [doc for doc in documents if combined(doc)]


def create_sample_documents() -> List[Document]:
    """Create sample documents with credibility metadata."""
    return [
        Document(
            "Breakthrough in quantum computing",
            {
                "source": "Nature",
                "source_type": "peer_reviewed",
                "credibility_score": 0.98,
                "citation_count": 150,
                "verified": True,
                "year": 2024,
            },
        ),
        Document(
            "AI ethics framework proposal",
            {
                "source": "IEEE",
                "source_type": "academic",
                "credibility_score": 0.90,
                "citation_count": 45,
                "verified": True,
                "year": 2024,
            },
        ),
        Document(
            "Market analysis report",
            {
                "source": "Reuters",
                "source_type": "news",
                "credibility_score": 0.80,
                "citation_count": 12,
                "verified": True,
                "year": 2024,
            },
        ),
        Document(
            "Personal thoughts on AI",
            {
                "source": "Personal Blog",
                "source_type": "blog",
                "credibility_score": 0.25,
                "citation_count": 0,
                "verified": False,
                "year": 2023,
            },
        ),
        Document(
            "Opinion: The future of work",
            {
                "source": "Medium",
                "source_type": "blog",
                "credibility_score": 0.40,
                "citation_count": 3,
                "verified": False,
                "year": 2024,
            },
        ),
        Document(
            "Peer-reviewed climate study",
            {
                "source": "Science",
                "source_type": "peer_reviewed",
                "credibility_score": 0.97,
                "citation_count": 200,
                "verified": True,
                "year": 2024,
            },
        ),
        Document(
            "Viral social media post",
            {
                "source": "Twitter",
                "source_type": "social_media",
                "credibility_score": 0.15,
                "citation_count": 0,
                "verified": False,
                "year": 2024,
            },
        ),
    ]


def demonstrate_basic_credibility():
    """Demonstrate basic credibility score filtering."""
    print("\n" + "=" * 60)
    print("1. HIGH CREDIBILITY ONLY: Score >= 0.8")
    print("=" * 60)

    docs = create_sample_documents()
    cf = CredibilityFilter

    filtered = cf.apply_filters(docs, cf.min_score(0.8))

    for doc in filtered:
        score = doc.metadata["credibility_score"]
        print(f"  [Score: {score}] {doc.metadata['source']}: {doc.content[:60]}...")

    print(f"\n✅ {len(filtered)}/{len(docs)} documents meet high credibility threshold")


def demonstrate_source_filtering():
    """Demonstrate source-based filtering."""
    print("\n" + "=" * 60)
    print("2. TRUSTED SOURCES: Nature, Science, Reuters only")
    print("=" * 60)

    docs = create_sample_documents()
    cf = CredibilityFilter

    filtered = cf.apply_filters(
        docs, cf.allowed_sources(["Nature", "Science", "Reuters"])
    )

    for doc in filtered:
        print(f"  [{doc.metadata['source']}] {doc.content[:60]}...")

    print(f"\n✅ {len(filtered)}/{len(docs)} documents from trusted sources")


def demonstrate_exclusion_filtering():
    """Demonstrate source exclusion."""
    print("\n" + "=" * 60)
    print("3. EXCLUDE UNRELIABLE: No blogs or social media")
    print("=" * 60)

    docs = create_sample_documents()
    cf = CredibilityFilter

    filtered = cf.apply_filters(
        docs, cf.excluded_sources(["Personal Blog", "Medium", "Twitter"])
    )

    for doc in filtered:
        source_type = doc.metadata["source_type"]
        print(f"  [{source_type}] {doc.metadata['source']}: {doc.content[:60]}...")

    print(
        f"\n✅ {len(filtered)}/{len(docs)} documents after excluding unreliable sources"
    )


def demonstrate_combined_credibility():
    """Demonstrate combined credibility filtering."""
    print("\n" + "=" * 60)
    print("4. COMBINED: High score + verified + recent + cited")
    print("=" * 60)

    docs = create_sample_documents()
    cf = CredibilityFilter

    filtered = cf.apply_filters(
        docs, cf.min_score(0.8), cf.verified_only(), cf.min_citations(10)
    )

    for doc in filtered:
        print(
            f"  [Score: {doc.metadata['credibility_score']}] "
            f"[Citations: {doc.metadata['citation_count']}] "
            f"{doc.metadata['source']}: {doc.content[:50]}..."
        )

    print(f"\n✅ {len(filtered)}/{len(docs)} documents meet all credibility criteria")


def demonstrate_scoring_calculation():
    """Demonstrate the credibility scoring system."""
    print("\n" + "=" * 60)
    print("5. CREDIBILITY SCORING DEMONSTRATION")
    print("=" * 60)

    cs = CredibilityScorer

    test_cases = [
        ("Nature", 200, 30, True),
        ("IEEE", 50, 180, True),
        ("Reuters", 15, 60, True),
        ("Medium", 3, 45, False),
        ("Personal Blog", 0, 365, False),
    ]

    for source, citations, days, verified in test_cases:
        score = cs.calculate_document_credibility(source, citations, days, verified)
        tier = cs.get_credibility_tier(score)
        print(f"\n  {source}:")
        print(f"    Citations: {citations}, Age: {days}d, Verified: {verified}")
        print(f"    Score: {score} → Tier: {tier.value}")


def main():
    """Run all credibility filtering demonstrations."""
    print("=" * 60)
    print("SOURCE CREDIBILITY FILTERING DEMONSTRATION")
    print("=" * 60)

    demonstrate_basic_credibility()
    demonstrate_source_filtering()
    demonstrate_exclusion_filtering()
    demonstrate_combined_credibility()
    demonstrate_scoring_calculation()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Credibility filtering combines source reputation,")
    print("   citation analysis, recency, and verification status.")


if __name__ == "__main__":
    main()
