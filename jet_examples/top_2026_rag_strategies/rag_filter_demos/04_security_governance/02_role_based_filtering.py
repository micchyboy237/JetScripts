"""
02_role_based_filtering.py

Role-based dynamic filtering for RAG systems.
Demonstrates generating different filters based on user roles.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class UserRole(str, Enum):
    """User roles with different access levels."""

    ADMIN = "admin"
    DIRECTOR = "director"
    MANAGER = "manager"
    SENIOR = "senior"
    EMPLOYEE = "employee"
    CONTRACTOR = "contractor"
    INTERN = "intern"
    GUEST = "guest"


class VisibilityLevel(str, Enum):
    """Content visibility levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    TEAM = "team"
    MANAGEMENT = "management"
    EXECUTIVE = "executive"


@dataclass
class RoleConfig:
    """Configuration for each role's access patterns."""

    role: UserRole
    visibility_levels: List[VisibilityLevel]
    max_document_age_days: Optional[int]
    allowed_departments: List[str]
    can_see_drafts: bool
    can_see_archived: bool
    min_credibility_score: float
    rate_limit_per_minute: int


@dataclass
class Document:
    """Document with visibility metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoleBasedFilterEngine:
    """
    Generate filters based on user roles.
    Each role has predefined access patterns and restrictions.
    """

    # Role configurations
    ROLE_CONFIGS = {
        UserRole.ADMIN: RoleConfig(
            role=UserRole.ADMIN,
            visibility_levels=list(VisibilityLevel),
            max_document_age_days=None,
            allowed_departments=["*"],
            can_see_drafts=True,
            can_see_archived=True,
            min_credibility_score=0.0,
            rate_limit_per_minute=1000,
        ),
        UserRole.DIRECTOR: RoleConfig(
            role=UserRole.DIRECTOR,
            visibility_levels=[
                VisibilityLevel.PUBLIC,
                VisibilityLevel.INTERNAL,
                VisibilityLevel.TEAM,
                VisibilityLevel.MANAGEMENT,
                VisibilityLevel.EXECUTIVE,
            ],
            max_document_age_days=1095,  # 3 years
            allowed_departments=["*"],
            can_see_drafts=True,
            can_see_archived=False,
            min_credibility_score=0.3,
            rate_limit_per_minute=500,
        ),
        UserRole.MANAGER: RoleConfig(
            role=UserRole.MANAGER,
            visibility_levels=[
                VisibilityLevel.PUBLIC,
                VisibilityLevel.INTERNAL,
                VisibilityLevel.TEAM,
                VisibilityLevel.MANAGEMENT,
            ],
            max_document_age_days=730,  # 2 years
            allowed_departments=["Technology", "Engineering", "Product"],
            can_see_drafts=True,
            can_see_archived=False,
            min_credibility_score=0.5,
            rate_limit_per_minute=200,
        ),
        UserRole.EMPLOYEE: RoleConfig(
            role=UserRole.EMPLOYEE,
            visibility_levels=[
                VisibilityLevel.PUBLIC,
                VisibilityLevel.INTERNAL,
                VisibilityLevel.TEAM,
            ],
            max_document_age_days=365,
            allowed_departments=["Technology", "Engineering"],
            can_see_drafts=False,
            can_see_archived=False,
            min_credibility_score=0.6,
            rate_limit_per_minute=100,
        ),
        UserRole.CONTRACTOR: RoleConfig(
            role=UserRole.CONTRACTOR,
            visibility_levels=[VisibilityLevel.PUBLIC, VisibilityLevel.INTERNAL],
            max_document_age_days=180,
            allowed_departments=["Technology"],
            can_see_drafts=False,
            can_see_archived=False,
            min_credibility_score=0.7,
            rate_limit_per_minute=60,
        ),
        UserRole.INTERN: RoleConfig(
            role=UserRole.INTERN,
            visibility_levels=[VisibilityLevel.PUBLIC],
            max_document_age_days=365,
            allowed_departments=["Technology"],
            can_see_drafts=False,
            can_see_archived=False,
            min_credibility_score=0.5,
            rate_limit_per_minute=30,
        ),
        UserRole.GUEST: RoleConfig(
            role=UserRole.GUEST,
            visibility_levels=[VisibilityLevel.PUBLIC],
            max_document_age_days=365,
            allowed_departments=[],
            can_see_drafts=False,
            can_see_archived=False,
            min_credibility_score=0.8,
            rate_limit_per_minute=10,
        ),
    }

    @classmethod
    def get_role_config(cls, role: UserRole) -> RoleConfig:
        """Get configuration for a role."""
        return cls.ROLE_CONFIGS.get(role, cls.ROLE_CONFIGS[UserRole.GUEST])

    @classmethod
    def build_chromadb_filter(cls, role: UserRole) -> Dict:
        """Build ChromaDB filter for a role."""
        config = cls.get_role_config(role)
        conditions = []

        # Visibility filter
        visibility_values = [v.value for v in config.visibility_levels]
        conditions.append({"visibility": {"$in": visibility_values}})

        # Department filter
        if config.allowed_departments != ["*"] and config.allowed_departments:
            conditions.append({"department": {"$in": config.allowed_departments}})

        # Age filter
        if config.max_document_age_days is not None:
            cutoff = datetime.now() - timedelta(days=config.max_document_age_days)
            conditions.append({"created_at": {"$gte": cutoff.isoformat()}})

        # Drafts filter
        if not config.can_see_drafts:
            conditions.append({"status": {"$ne": "draft"}})

        # Archived filter
        if not config.can_see_archived:
            conditions.append({"status": {"$ne": "archived"}})

        # Credibility filter
        if config.min_credibility_score > 0:
            conditions.append(
                {"credibility_score": {"$gte": config.min_credibility_score}}
            )

        return {"$and": conditions} if len(conditions) > 1 else conditions[0]

    @classmethod
    def build_generic_filters(cls, role: UserRole) -> List[Callable]:
        """Build generic Python filter functions for a role."""
        config = cls.get_role_config(role)
        filters = []

        # Visibility
        visibility_values = [v.value for v in config.visibility_levels]
        filters.append(
            lambda meta, vis=visibility_values: meta.get("visibility") in vis
        )

        # Department
        if config.allowed_departments != ["*"] and config.allowed_departments:
            allowed = config.allowed_departments
            filters.append(lambda meta, depts=allowed: meta.get("department") in depts)

        # Age
        if config.max_document_age_days is not None:
            cutoff = datetime.now() - timedelta(days=config.max_document_age_days)
            filters.append(
                lambda meta, cut=cutoff: datetime.fromisoformat(
                    meta.get("created_at", "2000-01-01")
                )
                >= cut
            )

        # Drafts
        if not config.can_see_drafts:
            filters.append(lambda meta: meta.get("status") != "draft")

        # Archived
        if not config.can_see_archived:
            filters.append(lambda meta: meta.get("status") != "archived")

        # Credibility
        if config.min_credibility_score > 0:
            min_score = config.min_credibility_score
            filters.append(
                lambda meta, score=min_score: meta.get("credibility_score", 0) >= score
            )

        return filters


def demonstrate_role_filters():
    """Demonstrate filter generation for different roles."""
    print("\n" + "=" * 60)
    print("1. ROLE-BASED FILTER GENERATION")
    print("=" * 60)

    roles = [
        UserRole.ADMIN,
        UserRole.MANAGER,
        UserRole.EMPLOYEE,
        UserRole.INTERN,
    ]

    for role in roles:
        config = RoleBasedFilterEngine.get_role_config(role)
        chroma_filter = RoleBasedFilterEngine.build_chromadb_filter(role)

        print(f"\n  {role.value.upper()}:")
        print(f"    Visibility: {[v.value for v in config.visibility_levels]}")
        print(f"    Max Age: {config.max_document_age_days or 'Unlimited'} days")
        print(f"    Departments: {config.allowed_departments}")
        print(f"    Drafts: {'✓' if config.can_see_drafts else '✗'}")
        print(f"    Min Credibility: {config.min_credibility_score}")

        import json

        print(f"    Filter: {json.dumps(chroma_filter)}")

    print("\n✅ Role-based filters generated")


def demonstrate_document_filtering_by_role():
    """Demonstrate filtering documents by role."""
    print("\n" + "=" * 60)
    print("2. DOCUMENT ACCESS BY ROLE")
    print("=" * 60)

    from datetime import datetime, timedelta

    docs = [
        Document(
            "Public blog post",
            {
                "visibility": "public",
                "department": "Technology",
                "status": "published",
                "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
                "credibility_score": 0.9,
            },
        ),
        Document(
            "Internal design doc",
            {
                "visibility": "internal",
                "department": "Technology",
                "status": "published",
                "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "credibility_score": 0.8,
            },
        ),
        Document(
            "Management strategy",
            {
                "visibility": "management",
                "department": "Product",
                "status": "draft",
                "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
                "credibility_score": 0.7,
            },
        ),
        Document(
            "Executive memo",
            {
                "visibility": "executive",
                "department": "Executive",
                "status": "published",
                "created_at": (datetime.now() - timedelta(days=400)).isoformat(),
                "credibility_score": 0.95,
            },
        ),
        Document(
            "Old archived project",
            {
                "visibility": "internal",
                "department": "Technology",
                "status": "archived",
                "created_at": (datetime.now() - timedelta(days=800)).isoformat(),
                "credibility_score": 0.5,
            },
        ),
    ]

    roles_to_check = [UserRole.ADMIN, UserRole.EMPLOYEE, UserRole.INTERN]

    print("\n  Document access matrix:")
    header = f"  {'Document':<30}"
    for role in roles_to_check:
        header += f" {role.value:<12}"
    print(header)
    print(f"  {'-' * 66}")

    for doc in docs:
        content = doc.content[:25]
        row = f"  {content:<30}"

        for role in roles_to_check:
            filters = RoleBasedFilterEngine.build_generic_filters(role)
            can_access = all(f(doc.metadata) for f in filters)
            row += f" {'✓' if can_access else '✗':<12}"

        print(row)

    print("\n✅ Document access matrix by role")


def demonstrate_rate_limiting():
    """Demonstrate role-based rate limiting."""
    print("\n" + "=" * 60)
    print("3. ROLE-BASED RATE LIMITING")
    print("=" * 60)

    for role in UserRole:
        config = RoleBasedFilterEngine.get_role_config(role)
        print(f"\n  {role.value}:")
        print(f"    Max requests/minute: {config.rate_limit_per_minute}")
        print(f"    Max requests/second: {config.rate_limit_per_minute / 60:.1f}")

    print("\n✅ Rate limits defined for all roles")


def demonstrate_dynamic_role_escalation():
    """Demonstrate dynamic permission escalation."""
    print("\n" + "=" * 60)
    print("4. DYNAMIC ROLE ESCALATION PATTERN")
    print("=" * 60)

    class EscalatingFilterBuilder:
        """Allows temporary permission escalation with audit trail."""

        def __init__(self, base_role: UserRole):
            self.base_role = base_role
            self.escalated_role = None
            self.escalation_reason = None
            self.escalation_expiry = None

        def escalate(
            self, target_role: UserRole, reason: str, duration_minutes: int = 30
        ):
            """Temporarily escalate permissions."""
            self.escalated_role = target_role
            self.escalation_reason = reason
            self.escalation_expiry = datetime.now() + timedelta(
                minutes=duration_minutes
            )
            print(f"  ⚠️  Escalated from {self.base_role.value} to {target_role.value}")
            print(f"     Reason: {reason}")
            print(f"     Expires: {self.escalation_expiry.strftime('%H:%M:%S')}")

        def get_effective_role(self) -> UserRole:
            """Get current effective role."""
            if (
                self.escalated_role
                and self.escalation_expiry
                and datetime.now() < self.escalation_expiry
            ):
                return self.escalated_role
            return self.base_role

        def build_filter(self) -> Dict:
            """Build filter for current effective role."""
            effective_role = self.get_effective_role()
            base_filter = RoleBasedFilterEngine.build_chromadb_filter(effective_role)

            # Add audit metadata
            base_filter["_audit"] = {
                "base_role": self.base_role.value,
                "effective_role": effective_role.value,
                "escalated": self.escalated_role is not None,
                "reason": self.escalation_reason,
            }

            return base_filter

    # Demonstrate escalation
    builder = EscalatingFilterBuilder(UserRole.EMPLOYEE)

    print(f"\n  Base role: {builder.base_role.value}")
    print(f"  Effective role: {builder.get_effective_role().value}")

    builder.escalate(
        UserRole.MANAGER, "Need access to management reports for Q4 review"
    )

    filter_with_audit = builder.build_filter()
    print("\n  Filter with audit trail:")
    import json

    print(f"  {json.dumps(filter_with_audit, indent=4)}")

    print("\n✅ Dynamic role escalation with audit")


def main():
    """Run all role-based filtering demonstrations."""
    print("=" * 60)
    print("ROLE-BASED DYNAMIC FILTERING")
    print("=" * 60)

    demonstrate_role_filters()
    demonstrate_document_filtering_by_role()
    demonstrate_rate_limiting()
    demonstrate_dynamic_role_escalation()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Role-based filtering ensures users see only")
    print("   content appropriate for their access level.")


if __name__ == "__main__":
    main()
