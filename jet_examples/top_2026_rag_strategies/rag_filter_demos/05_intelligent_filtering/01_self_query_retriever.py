"""
01_self_query_retriever.py

LangChain Self-Query Retriever demonstration.
Automatically converts natural language queries into structured filters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AttributeInfo:
    """Metadata field definition for self-query retriever."""

    name: str
    description: str
    type: str


@dataclass
class Document:
    """Document with metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelfQuerySimulator:
    """
    Simulates the Self-Query Retriever's filter extraction logic.
    In production, this would use LangChain's SelfQueryRetriever with an LLM.
    """

    def __init__(self, metadata_fields: List[AttributeInfo], llm_model: str = "gpt-4"):
        self.metadata_fields = metadata_fields
        self.llm_model = llm_model

        # Build field descriptions for prompt
        self.field_descriptions = self._build_field_descriptions()

    def _build_field_descriptions(self) -> str:
        """Build field descriptions for the LLM prompt."""
        descriptions = []
        for field in self.metadata_fields:
            descriptions.append(f"  - {field.name} ({field.type}): {field.description}")
        return "\n".join(descriptions)

    def _extract_filters(self, query: str) -> Dict:
        """
        Simulate LLM-based filter extraction.

        In production, this would call:
        ```python
        from langchain.retrievers.self_query.base import SelfQueryRetriever
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4", temperature=0)
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            enable_limit=True,
        )
        results = retriever.invoke(query)
        ```
        """

        # Simulated extraction rules for demonstration
        query_lower = query.lower()
        filters = {}

        # Department extraction
        departments = ["technology", "hr", "finance", "legal", "engineering", "product"]
        for dept in departments:
            if dept in query_lower:
                filters["department"] = dept.title()
                break

        # Year extraction
        import re

        year_pattern = r"\b(20\d{2})\b"
        years = re.findall(year_pattern, query)
        if len(years) == 1:
            filters["year"] = int(years[0])
        elif len(years) >= 2:
            filters["min_year"] = min(int(y) for y in years)
            filters["max_year"] = max(int(y) for y in years)

        # Topic extraction
        topics = ["ai", "machine learning", "cloud", "security", "compliance", "data"]
        found_topics = [t for t in topics if t in query_lower]
        if found_topics:
            filters["topic"] = found_topics

        # Document type extraction
        doc_types = ["report", "policy", "memo", "whitepaper", "guide", "strategy"]
        found_types = [t for t in doc_types if t in query_lower]
        if found_types:
            filters["doc_type"] = found_types

        return filters

    def build_chromadb_filter(self, extracted_filters: Dict) -> Dict:
        """Convert extracted filters to ChromaDB format."""
        conditions = []

        if "department" in extracted_filters:
            conditions.append({"department": extracted_filters["department"]})

        if "year" in extracted_filters:
            conditions.append({"year": {"$eq": extracted_filters["year"]}})
        else:
            if "min_year" in extracted_filters:
                conditions.append({"year": {"$gte": extracted_filters["min_year"]}})
            if "max_year" in extracted_filters:
                conditions.append({"year": {"$lte": extracted_filters["max_year"]}})

        if "topic" in extracted_filters:
            topics = extracted_filters["topic"]
            if isinstance(topics, list) and len(topics) == 1:
                conditions.append({"topic": topics[0]})
            elif isinstance(topics, list):
                conditions.append({"topic": {"$in": topics}})

        if "doc_type" in extracted_filters:
            types = extracted_filters["doc_type"]
            if isinstance(types, list) and len(types) == 1:
                conditions.append({"doc_type": types[0]})
            elif isinstance(types, list):
                conditions.append({"doc_type": {"$in": types}})

        if len(conditions) == 0:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def query(self, natural_language_query: str) -> Dict:
        """Process a natural language query and return filtered results."""
        print(f"\n  Query: '{natural_language_query}'")

        # Extract filters
        extracted = self._extract_filters(natural_language_query)
        print(f"  Extracted filters: {extracted}")

        # Build database filter
        db_filter = self.build_chromadb_filter(extracted)
        print(f"  Database filter: {db_filter}")

        return {
            "query": natural_language_query,
            "extracted_filters": extracted,
            "db_filter": db_filter,
        }


def demonstrate_self_query_basics():
    """Demonstrate basic self-query capabilities."""
    print("\n" + "=" * 60)
    print("1. SELF-QUERY: Natural Language to Filters")
    print("=" * 60)

    metadata_fields = [
        AttributeInfo("department", "The department the document belongs to", "string"),
        AttributeInfo("year", "The publication year of the document", "integer"),
        AttributeInfo("topic", "The main topic of the document", "string"),
        AttributeInfo("doc_type", "Type of document (report, policy, memo)", "string"),
        AttributeInfo("credibility_score", "Source credibility (0-1)", "float"),
    ]

    retriever = SelfQuerySimulator(metadata_fields)

    queries = [
        "What were the technology department's AI initiatives in 2024?",
        "Show me finance reports from 2023",
        "Find cloud migration strategies",
        "What policies were updated in 2023 and 2024?",
    ]

    for query in queries:
        result = retriever.query(query)

    print("\n✅ Self-query demonstrations complete")


def demonstrate_complex_queries():
    """Demonstrate complex query understanding."""
    print("\n" + "=" * 60)
    print("2. COMPLEX QUERY UNDERSTANDING")
    print("=" * 60)

    metadata_fields = [
        AttributeInfo("department", "Document department", "string"),
        AttributeInfo("year", "Publication year", "integer"),
        AttributeInfo("author", "Document author", "string"),
        AttributeInfo("status", "Document status", "string"),
    ]

    retriever = SelfQuerySimulator(metadata_fields)

    complex_queries = [
        "Find recent compliance documents from the legal department in 2024",
        "What did Alice write about machine learning?",
        "Show me published strategy documents about cloud migration",
    ]

    for query in complex_queries:
        result = retriever.query(query)

    print("\n✅ Complex query understanding complete")


def demonstrate_limit_and_pagination():
    """Demonstrate limit handling in self-query."""
    print("\n" + "=" * 60)
    print("3. LIMIT HANDLING")
    print("=" * 60)

    print("\n  Self-query retrievers support automatic limit extraction:")
    print("  - 'Show me top 5 AI papers' → limit=5")
    print("  - 'Find the best 10 matches' → limit=10")
    print("  - 'Give me 3 examples' → limit=3")
    print("\n  The enable_limit=True parameter allows the LLM")
    print("  to extract result count limits from queries.")
    print("\n✅ Limit handling explained")


def demonstrate_production_setup():
    """Demonstrate production Self-Query Retriever setup."""
    print("\n" + "=" * 60)
    print("4. PRODUCTION SETUP GUIDE")
    print("=" * 60)

    setup_code = """
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# 1. Define metadata fields
metadata_field_info = [
    AttributeInfo(
        name="department",
        description="The department the document belongs to",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The publication year of the document",
        type="integer",
    ),
    AttributeInfo(
        name="topic",
        description="The main topic of the document",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="The author of the document",
        type="string",
    ),
    AttributeInfo(
        name="doc_type",
        description="Type of document (report, policy, memo)",
        type="string",
    ),
    AttributeInfo(
        name="credibility_score",
        description="Source credibility (0-1)",
        type="float",
    ),
]

# 2. Document content description
document_content_description = (
    "Company internal documents including reports, "
    "policies, and technical documentation"
)

# 3. Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 4. Create self-query retriever
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True,
)

# 5. Natural language query (filters auto-extracted)
results = self_query_retriever.invoke(
    "What were the technology department's AI initiatives in 2024?"
)
# Automatically extracts: 
# {"department": "Technology", "year": 2024, "topic": "AI"}
"""

    print("\n  Production setup pattern:")
    print(f"  {setup_code}")
    print("\n✅ Production setup guide complete")


def main():
    """Run all self-query retriever demonstrations."""
    print("=" * 60)
    print("SELF-QUERY RETRIEVER DEMONSTRATION")
    print("=" * 60)
    print("\nNote: Demonstrates the self-query pattern.")
    print("Production use requires LangChain and an LLM.\n")

    demonstrate_self_query_basics()
    demonstrate_complex_queries()
    demonstrate_limit_and_pagination()
    demonstrate_production_setup()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Self-query retrievers automatically convert")
    print("   natural language into structured filters.")


if __name__ == "__main__":
    main()
