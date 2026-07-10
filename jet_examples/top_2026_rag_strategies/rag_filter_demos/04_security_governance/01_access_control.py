"""
01_access_control.py

Row-level security and access control for RAG systems.
Demonstrates user-specific filtering based on permissions.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional


class SecurityClearance(str, Enum):
    """Security clearance levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class Department(str, Enum):
    """Departments for access control."""

    TECHNOLOGY = "Technology"
    ENGINEERING = "Engineering"
    PRODUCT = "Product"
    HR = "HR"
    FINANCE = "Finance"
    LEGAL = "Legal"
    EXECUTIVE = "Executive"


@dataclass
class UserProfile:
    """User profile with permissions."""

    user_id: str
    username: str
    role: str
    allowed_departments: List[str]
    security_clearance: SecurityClearance
    allowed_projects: List[str]
    excluded_projects: List[str] = field(default_factory=list)
    team_ids: List[str] = field(default_factory=list)


@dataclass
class Document:
    """Document with security metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AccessControlSystem:
    """
    Manage user permissions and build security filters.
    In production, this would connect to your auth/IAM system.
    """

    # Simulated user database
    USERS = {
        "user_001": UserProfile(
            "user_001",
            "alice",
            "admin",
            ["Technology", "Engineering", "Product", "Executive"],
            SecurityClearance.TOP_SECRET,
            ["proj_001", "proj_002", "proj_003"],
        ),
        "user_002": UserProfile(
            "user_002",
            "bob",
            "manager",
            ["Technology", "Engineering"],
            SecurityClearance.CONFIDENTIAL,
            ["proj_001", "proj_002"],
            excluded_projects=["proj_secret"],
        ),
        "user_003": UserProfile(
            "user_003",
            "carol",
            "employee",
            ["Technology"],
            SecurityClearance.INTERNAL,
            ["proj_001"],
        ),
        "user_004": UserProfile(
            "user_004",
            "dave",
            "intern",
            ["Technology"],
            SecurityClearance.PUBLIC,
            [],
            team_ids=["team_interns"],
        ),
    }

    CLEARANCE_LEVELS = {
        SecurityClearance.PUBLIC: 0,
        SecurityClearance.INTERNAL: 1,
        SecurityClearance.CONFIDENTIAL: 2,
        SecurityClearance.SECRET: 3,
        SecurityClearance.TOP_SECRET: 4,
    }

    @classmethod
    @lru_cache(maxsize=1000)
    def get_user(cls, user_id: str) -> Optional[UserProfile]:
        """Get user profile (cached)."""
        return cls.USERS.get(user_id)

    @classmethod
    def get_user_permissions(cls, user_id: str) -> Dict[str, Any]:
        """Get user's effective permissions."""
        user = cls.get_user(user_id)
        if not user:
            return {
                "allowed_departments": [],
                "security_clearance": SecurityClearance.PUBLIC,
                "allowed_projects": [],
                "excluded_projects": [],
            }

        return {
            "allowed_departments": user.allowed_departments,
            "security_clearance": user.security_clearance,
            "allowed_projects": user.allowed_projects,
            "excluded_projects": user.excluded_projects,
        }

    @classmethod
    def can_access_document(cls, user_id: str, document: Document) -> bool:
        """Check if user can access a specific document."""
        user = cls.get_user(user_id)
        if not user:
            return False

        doc_meta = document.metadata

        # Check department access
        if doc_meta.get("department") not in user.allowed_departments:
            return False

        # Check clearance level
        doc_clearance = doc_meta.get("clearance_level", SecurityClearance.PUBLIC)
        user_level = cls.CLEARANCE_LEVELS.get(user.security_clearance, 0)
        doc_level = cls.CLEARANCE_LEVELS.get(doc_clearance, 0)

        if doc_level > user_level:
            return False

        # Check project access
        doc_project = doc_meta.get("project")
        if doc_project:
            if doc_project in user.excluded_projects:
                return False
            if (
                doc_project not in user.allowed_projects
                and "*" not in user.allowed_projects
            ):
                return False

        return True


class SecurityFilterBuilder:
    """Build security filters for vector database queries."""

    @staticmethod
    def build_chromadb_filter(user_id: str) -> Dict:
        """Build ChromaDB-compatible security filter."""
        perms = AccessControlSystem.get_user_permissions(user_id)

        if not perms["allowed_departments"]:
            return {"visibility": {"$eq": "public"}}

        conditions = [
            {"department": {"$in": perms["allowed_departments"]}},
            {
                "clearance_level": {
                    "$in": SecurityFilterBuilder._get_clearance_levels(
                        perms["security_clearance"]
                    )
                }
            },
        ]

        if perms["allowed_projects"]:
            conditions.append({"project": {"$in": perms["allowed_projects"]}})

        if perms["excluded_projects"]:
            for proj in perms["excluded_projects"]:
                conditions.append({"project": {"$ne": proj}})

        return {"$and": conditions}

    @staticmethod
    def _get_clearance_levels(max_clearance: SecurityClearance) -> List[str]:
        """Get all clearance levels up to max."""
        levels = list(SecurityClearance)
        max_index = levels.index(max_clearance)
        return [l.value for l in levels[: max_index + 1]]

    @staticmethod
    def build_pinecone_filter(user_id: str) -> Dict:
        """Build Pinecone-compatible security filter."""
        perms = AccessControlSystem.get_user_permissions(user_id)

        return {
            "$and": [
                {"department": {"$in": perms["allowed_departments"]}},
                {"clearance_level": {"$lte": perms["security_clearance"].value}},
                {"project": {"$in": perms["allowed_projects"]}},
                {"project": {"$nin": perms["excluded_projects"]}},
            ]
        }


def create_sample_documents() -> List[Document]:
    """Create sample documents with security metadata."""
    return [
        Document(
            "Public product roadmap",
            {
                "department": "Product",
                "clearance_level": SecurityClearance.PUBLIC,
                "project": "proj_001",
                "visibility": "public",
            },
        ),
        Document(
            "Internal engineering design",
            {
                "department": "Engineering",
                "clearance_level": SecurityClearance.INTERNAL,
                "project": "proj_002",
                "visibility": "internal",
            },
        ),
        Document(
            "Confidential financial projections",
            {
                "department": "Finance",
                "clearance_level": SecurityClearance.CONFIDENTIAL,
                "project": "proj_003",
                "visibility": "management",
            },
        ),
        Document(
            "Secret acquisition strategy",
            {
                "department": "Executive",
                "clearance_level": SecurityClearance.SECRET,
                "project": "proj_secret",
                "visibility": "management",
            },
        ),
        Document(
            "Technology blog post",
            {
                "department": "Technology",
                "clearance_level": SecurityClearance.PUBLIC,
                "project": "proj_001",
                "visibility": "public",
            },
        ),
        Document(
            "HR policy document",
            {
                "department": "HR",
                "clearance_level": SecurityClearance.INTERNAL,
                "project": None,
                "visibility": "internal",
            },
        ),
    ]


def demonstrate_user_access_check():
    """Demonstrate per-document access checking."""
    print("\n" + "=" * 60)
    print("1. PER-DOCUMENT ACCESS CHECK")
    print("=" * 60)

    docs = create_sample_documents()
    users_to_check = ["user_001", "user_002", "user_003", "user_004"]

    print("\n  Documents and user access:")
    print(
        f"  {'Document':<40} {'Admin':<8} {'Manager':<8} {'Employee':<8} {'Intern':<8}"
    )
    print(f"  {'-' * 72}")

    for doc in docs:
        content = doc.content[:35]
        access = []
        for uid in users_to_check:
            can_access = AccessControlSystem.can_access_document(uid, doc)
            access.append("✓" if can_access else "✗")

        print(
            f"  {content:<40} {access[0]:<8} {access[1]:<8} {access[2]:<8} {access[3]:<8}"
        )

    print("\n✅ Access control matrix complete")


def demonstrate_security_filter_building():
    """Demonstrate building security filters for different users."""
    print("\n" + "=" * 60)
    print("2. SECURITY FILTER CONSTRUCTION")
    print("=" * 60)

    users = [
        ("user_001", "Admin (Full Access)"),
        ("user_003", "Employee (Limited Access)"),
        ("user_004", "Intern (Minimal Access)"),
    ]

    for user_id, description in users:
        print(f"\n  {description}:")
        chroma_filter = SecurityFilterBuilder.build_chromadb_filter(user_id)
        import json

        print(f"    ChromaDB filter: {json.dumps(chroma_filter, indent=6)}")

    print("\n✅ Security filters generated for all user types")


def demonstrate_filter_merging():
    """Demonstrate merging security filters with query filters."""
    print("\n" + "=" * 60)
    print("3. MERGING SECURITY + QUERY FILTERS")
    print("=" * 60)

    user_id = "user_002"  # Manager with limited access
    security_filter = SecurityFilterBuilder.build_chromadb_filter(user_id)

    # User query filter (what they're searching for)
    query_filter = {
        "$and": [{"year": {"$gte": 2024}}, {"status": {"$eq": "published"}}]
    }

    # Merge filters - security always comes first
    merged_filter = {"$and": [security_filter, query_filter]}

    print("\n  User: Manager (Bob)")
    print("\n  Security filter (automatic):")
    import json

    print(f"  {json.dumps(security_filter, indent=4)}")
    print("\n  Query filter (user's search):")
    print(f"  {json.dumps(query_filter, indent=4)}")
    print("\n  Merged filter (applied to search):")
    print(f"  {json.dumps(merged_filter, indent=4)}")
    print("\n✅ Security and query filters merged correctly")


def demonstrate_row_level_security_class():
    """Demonstrate a production-style secure retriever class."""
    print("\n" + "=" * 60)
    print("4. PRODUCTION SECURE RETRIEVER PATTERN")
    print("=" * 60)

    class SecureRAGRetriever:
        """Production pattern for secure RAG retrieval."""

        def __init__(self, user_id: str):
            self.user_id = user_id
            self.permissions = AccessControlSystem.get_user_permissions(user_id)
            self.security_filter = SecurityFilterBuilder.build_chromadb_filter(user_id)

        def search(self, query: str, additional_filters: Dict = None) -> Dict:
            """Perform secure search with automatic security filtering."""

            # Combine security with optional additional filters
            if additional_filters:
                final_filter = {"$and": [self.security_filter, additional_filters]}
            else:
                final_filter = self.security_filter

            # This would call the actual vector store
            return {
                "query": query,
                "filter": final_filter,
                "user_id": self.user_id,
                "permissions": self.permissions,
            }

    retriever = SecureRAGRetriever("user_002")
    result = retriever.search(
        "What's our engineering roadmap?", additional_filters={"year": {"$gte": 2024}}
    )

    print("\n  Search result:")
    import json

    print(f"  {json.dumps(result, indent=4)}")
    print("\n✅ Security automatically applied to all searches")


def main():
    """Run all access control demonstrations."""
    print("=" * 60)
    print("ACCESS CONTROL & ROW-LEVEL SECURITY")
    print("=" * 60)

    demonstrate_user_access_check()
    demonstrate_security_filter_building()
    demonstrate_filter_merging()
    demonstrate_row_level_security_class()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Security filters must always be the first layer")
    print("   applied before any user-specified filtering.")


if __name__ == "__main__":
    main()
