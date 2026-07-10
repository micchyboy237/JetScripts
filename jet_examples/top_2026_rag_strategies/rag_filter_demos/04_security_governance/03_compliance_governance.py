"""
03_compliance_governance.py

Compliance and governance filters for regulated environments.
Demonstrates GDPR, HIPAA, SOX, and retention policy filtering.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict


class ComplianceStandard(str, Enum):
    """Compliance standards and regulations."""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    SOC2 = "soc2"
    CCPA = "ccpa"


class DataClassification(str, Enum):
    """Data classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information
    PCI = "pci"  # Payment Card Information


class Region(str, Enum):
    """Geographic regions for data residency."""

    US = "US"
    EU = "EU"
    EEA = "EEA"
    UK = "UK"
    APAC = "APAC"
    GLOBAL = "global"


@dataclass
class Document:
    """Document with compliance metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceRuleEngine:
    """
    Manage compliance rules for different regulatory standards.
    Each standard imposes specific filtering requirements.
    """

    # Compliance rule definitions
    COMPLIANCE_RULES = {
        ComplianceStandard.GDPR: {
            "data_types_excluded": [DataClassification.PII],
            "regions_allowed": [Region.EU, Region.EEA],
            "max_retention_days": 730,  # 2 years
            "require_consent": True,
            "require_anonymization": False,
            "allow_cross_border": False,
        },
        ComplianceStandard.HIPAA: {
            "data_types_excluded": [DataClassification.PHI],
            "regions_allowed": [Region.US],
            "max_retention_days": 2190,  # 6 years
            "require_consent": True,
            "require_anonymization": False,
            "allow_cross_border": False,
        },
        ComplianceStandard.SOX: {
            "data_types_excluded": [],
            "regions_allowed": [Region.US, Region.GLOBAL],
            "max_retention_days": 2555,  # 7 years
            "require_consent": False,
            "require_anonymization": False,
            "allow_cross_border": True,
        },
        ComplianceStandard.PCI_DSS: {
            "data_types_excluded": [DataClassification.PCI],
            "regions_allowed": [Region.GLOBAL],
            "max_retention_days": 365,
            "require_consent": True,
            "require_anonymization": True,
            "allow_cross_border": True,
        },
    }

    # Organization retention policies
    RETENTION_POLICIES = {
        "default": {
            "max_age_days": 730,
            "archive_after_days": 365,
            "delete_after_days": 1095,
        },
        "financial": {
            "max_age_days": 2555,  # 7 years for financial records
            "archive_after_days": 365,
            "delete_after_days": 3650,
        },
        "hr": {
            "max_age_days": 1825,  # 5 years for HR records
            "archive_after_days": 730,
            "delete_after_days": 2555,
        },
    }

    @classmethod
    def get_compliance_filters(
        cls, standard: ComplianceStandard, include_base_policies: bool = True
    ) -> Dict:
        """Get ChromaDB-compatible compliance filters."""
        rules = cls.COMPLIANCE_RULES.get(standard, {})
        conditions = []

        # Exclude restricted data types
        if rules.get("data_types_excluded"):
            excluded = [dt.value for dt in rules["data_types_excluded"]]
            for dtype in excluded:
                conditions.append({"data_type": {"$ne": dtype}})

        # Region restriction
        if rules.get("regions_allowed"):
            allowed_regions = [r.value for r in rules["regions_allowed"]]
            if Region.GLOBAL.value not in allowed_regions:
                conditions.append({"region": {"$in": allowed_regions}})

        # Retention period
        if rules.get("max_retention_days"):
            cutoff = datetime.now() - timedelta(days=rules["max_retention_days"])
            conditions.append({"created_at": {"$gte": cutoff.isoformat()}})

        # Consent requirement
        if rules.get("require_consent"):
            conditions.append({"has_consent": {"$eq": True}})

        # Cross-border restriction
        if not rules.get("allow_cross_border", True):
            conditions.append({"cross_border_allowed": {"$eq": True}})

        # Document status
        conditions.append({"document_status": {"$eq": "approved"}})

        return {"$and": conditions} if conditions else {}

    @classmethod
    def get_retention_filter(cls, document_category: str = "default") -> Dict:
        """Get retention policy filter."""
        policy = cls.RETENTION_POLICIES.get(
            document_category, cls.RETENTION_POLICIES["default"]
        )

        cutoff = datetime.now() - timedelta(days=policy["max_age_days"])

        return {
            "$and": [
                {"created_at": {"$gte": cutoff.isoformat()}},
                {"status": {"$ne": "deleted"}},
            ]
        }

    @classmethod
    def identify_documents_for_archival(
        cls, document_category: str = "default"
    ) -> Dict:
        """Identify documents ready for archival."""
        policy = cls.RETENTION_POLICIES.get(
            document_category, cls.RETENTION_POLICIES["default"]
        )

        archive_cutoff = datetime.now() - timedelta(days=policy["archive_after_days"])
        delete_cutoff = datetime.now() - timedelta(days=policy["delete_after_days"])

        return {
            "$or": [
                {
                    "$and": [
                        {"created_at": {"$lte": archive_cutoff.isoformat()}},
                        {"created_at": {"$gt": delete_cutoff.isoformat()}},
                        {"status": {"$ne": "archived"}},
                    ]
                },
                {
                    "$and": [
                        {"created_at": {"$lte": delete_cutoff.isoformat()}},
                        {"status": {"$ne": "deleted"}},
                    ]
                },
            ]
        }


class GovernedRAGSystem:
    """Production RAG system with compliance governance."""

    def __init__(self, default_compliance: ComplianceStandard = None):
        self.default_compliance = default_compliance
        self.governance_rules = ComplianceRuleEngine.COMPLIANCE_RULES
        self.retention_policies = ComplianceRuleEngine.RETENTION_POLICIES

    def compliant_search(
        self,
        query: str,
        compliance_type: ComplianceStandard = None,
        document_category: str = "default",
        k: int = 5,
    ) -> Dict:
        """Search with compliance filters applied."""

        # Determine compliance standard
        standard = compliance_type or self.default_compliance

        # Get compliance filters
        compliance_filter = {}
        if standard:
            compliance_filter = ComplianceRuleEngine.get_compliance_filters(standard)

        # Get retention filter
        retention_filter = ComplianceRuleEngine.get_retention_filter(document_category)

        # Merge filters
        filters = []
        if compliance_filter:
            filters.append(compliance_filter)
        if retention_filter:
            filters.append(retention_filter)

        final_filter = (
            {"$and": filters} if len(filters) > 1 else (filters[0] if filters else {})
        )

        return {
            "query": query,
            "compliance_standard": standard.value if standard else None,
            "filter": final_filter,
            "limit": k,
            "audit_trail": {
                "timestamp": datetime.now().isoformat(),
                "compliance_applied": standard is not None,
                "retention_applied": True,
            },
        }


def demonstrate_compliance_standards():
    """Demonstrate different compliance standard filters."""
    print("\n" + "=" * 60)
    print("1. COMPLIANCE STANDARD FILTERS")
    print("=" * 60)

    for standard in ComplianceStandard:
        filters = ComplianceRuleEngine.get_compliance_filters(standard)
        rules = ComplianceRuleEngine.COMPLIANCE_RULES[standard]

        print(f"\n  {standard.value.upper()}:")
        print(
            f"    Excluded data types: {[dt.value for dt in rules.get('data_types_excluded', [])]}"
        )
        print(
            f"    Allowed regions: {[r.value for r in rules.get('regions_allowed', [])]}"
        )
        print(f"    Max retention: {rules.get('max_retention_days')} days")
        print(f"    Requires consent: {rules.get('require_consent')}")
        print(f"    Cross-border allowed: {rules.get('allow_cross_border')}")

    print("\n✅ Compliance standards defined")


def demonstrate_retention_policies():
    """Demonstrate retention policy filtering."""
    print("\n" + "=" * 60)
    print("2. RETENTION POLICY FILTERS")
    print("=" * 60)

    for category, policy in ComplianceRuleEngine.RETENTION_POLICIES.items():
        print(f"\n  {category.upper()}:")
        print(
            f"    Max age: {policy['max_age_days']} days ({policy['max_age_days'] / 365:.1f} years)"
        )
        print(f"    Archive after: {policy['archive_after_days']} days")
        print(f"    Delete after: {policy['delete_after_days']} days")

        retention_filter = ComplianceRuleEngine.get_retention_filter(category)
        print(f"    Filter: {retention_filter}")

    print("\n✅ Retention policies defined")


def demonstrate_archival_detection():
    """Demonstrate detecting documents for archival."""
    print("\n" + "=" * 60)
    print("3. ARCHIVAL DETECTION")
    print("=" * 60)

    for category in ["default", "financial", "hr"]:
        archival_filter = ComplianceRuleEngine.identify_documents_for_archival(category)
        policy = ComplianceRuleEngine.RETENTION_POLICIES[category]

        print(f"\n  {category.upper()}:")
        print(f"    Archive cutoff: {policy['archive_after_days']} days")
        print(f"    Delete cutoff: {policy['delete_after_days']} days")
        print("    Archival filter:")
        import json

        print(f"    {json.dumps(archival_filter, indent=6)}")

    print("\n✅ Archival detection queries ready")


def demonstrate_governed_rag():
    """Demonstrate the governed RAG system."""
    print("\n" + "=" * 60)
    print("4. GOVERNED RAG SEARCH")
    print("=" * 60)

    rag = GovernedRAGSystem(default_compliance=ComplianceStandard.GDPR)

    # Search with GDPR compliance
    result = rag.compliant_search(
        query="customer data handling procedures",
        compliance_type=ComplianceStandard.GDPR,
        document_category="default",
    )

    print("\n  GDPR-compliant search:")
    import json

    print(f"  {json.dumps(result, indent=4)}")

    # Search with SOX compliance for financial documents
    result = rag.compliant_search(
        query="Q4 financial statements",
        compliance_type=ComplianceStandard.SOX,
        document_category="financial",
    )

    print("\n  SOX-compliant financial search:")
    print(f"  {json.dumps(result, indent=4)}")

    print("\n✅ Governed RAG search complete")


def demonstrate_compliance_matrix():
    """Demonstrate document compliance checking."""
    print("\n" + "=" * 60)
    print("5. DOCUMENT COMPLIANCE MATRIX")
    print("=" * 60)

    from datetime import datetime, timedelta

    docs = [
        Document(
            "EU customer data with PII",
            {
                "data_type": "pii",
                "region": "EU",
                "has_consent": True,
                "cross_border_allowed": False,
                "document_status": "approved",
                "created_at": (datetime.now() - timedelta(days=100)).isoformat(),
            },
        ),
        Document(
            "US medical records",
            {
                "data_type": "phi",
                "region": "US",
                "has_consent": True,
                "cross_border_allowed": False,
                "document_status": "approved",
                "created_at": (datetime.now() - timedelta(days=500)).isoformat(),
            },
        ),
        Document(
            "Payment card data",
            {
                "data_type": "pci",
                "region": "global",
                "has_consent": True,
                "cross_border_allowed": True,
                "document_status": "approved",
                "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
            },
        ),
    ]

    standards = [
        ComplianceStandard.GDPR,
        ComplianceStandard.HIPAA,
        ComplianceStandard.PCI_DSS,
    ]

    print(f"\n  {'Document':<30}", end="")
    for std in standards:
        print(f" {std.value:<10}", end="")
    print()
    print(f"  {'-' * 60}")

    for doc in docs:
        content = doc.content[:25]
        print(f"  {content:<30}", end="")

        for std in standards:
            filters = ComplianceRuleEngine.get_compliance_filters(
                std, include_base_policies=False
            )

            # Simple check: if filter is empty, it's allowed
            if not filters:
                print(f" {'✓':<10}", end="")
                continue

            # Check if document would be excluded
            would_exclude = False
            if "$and" in filters:
                for condition in filters["$and"]:
                    for field, value in condition.items():
                        if field == "data_type" and "$ne" in value:
                            if doc.metadata.get("data_type") == value["$ne"]:
                                would_exclude = True
                        elif field == "region" and "$in" in value:
                            if doc.metadata.get("region") not in value["$in"]:
                                would_exclude = True

            print(f" {'✗' if would_exclude else '✓':<10}", end="")

        print()

    print("\n✅ Compliance matrix complete")


def main():
    """Run all compliance governance demonstrations."""
    print("=" * 60)
    print("COMPLIANCE & GOVERNANCE FILTERING")
    print("=" * 60)

    demonstrate_compliance_standards()
    demonstrate_retention_policies()
    demonstrate_archival_detection()
    demonstrate_governed_rag()
    demonstrate_compliance_matrix()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Compliance filters ensure regulatory requirements")
    print("   are automatically enforced in all searches.")


if __name__ == "__main__":
    main()
