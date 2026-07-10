"""
02_custom_llm_extractor.py

Custom LLM-powered filter extraction with Pydantic models.
Demonstrates building your own intelligent filter extraction pipeline.
"""

import json
import re
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Department(str, Enum):
    TECHNOLOGY = "Technology"
    HR = "HR"
    FINANCE = "Finance"
    LEGAL = "Legal"
    MARKETING = "Marketing"
    ENGINEERING = "Engineering"
    PRODUCT = "Product"


class DocumentType(str, Enum):
    REPORT = "report"
    POLICY = "policy"
    MEMO = "memo"
    RESEARCH_PAPER = "research_paper"
    WHITEPAPER = "whitepaper"
    GUIDE = "guide"
    STRATEGY = "strategy"
    REVIEW = "review"


class FilterCriteria(BaseModel):
    """Schema for extracted filter criteria from user queries."""

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
    topics: Optional[List[str]] = Field(
        default=None, description="Topics or keywords to search for"
    )
    authors: Optional[List[str]] = Field(default=None, description="Authors to include")
    min_credibility: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum credibility score (0-1)"
    )
    exclude_drafts: bool = Field(
        default=False, description="Whether to exclude draft documents"
    )
    status_filter: Optional[str] = Field(
        default=None, description="Filter by document status"
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


class LLMFilterExtractor:
    """
    Extract structured filters from natural language queries using an LLM.

    In production, this would call an LLM API. Here we simulate the extraction
    with pattern matching to demonstrate the complete pipeline.
    """

    def __init__(self, llm_model: str = "gpt-4"):
        self.llm_model = llm_model
        self.criteria_schema = FilterCriteria

    def _build_extraction_prompt(self, query: str) -> str:
        """Build the LLM prompt for filter extraction."""
        prompt = f"""
Analyze the following user query and extract filtering criteria.

Query: "{query}"

Return ONLY a JSON object with these fields (use null if not specified):
- departments: array of department names
- exclude_departments: array of department names to exclude
- min_year: integer (minimum publication year)
- max_year: integer (maximum publication year)
- doc_types: array of document types
- topics: array of topic keywords
- authors: array of author names
- min_credibility: float (0-1 scale)
- exclude_drafts: boolean
- status_filter: string (e.g., "published", "approved")

Examples:
Query: "Show me technology reports from 2023 about AI"
Output: {{"departments": ["Technology"], "min_year": 2023, "max_year": 2023, 
         "topics": ["AI"], "doc_types": ["report"]}}

Query: "What did John Smith write about cloud computing, excluding drafts?"
Output: {{"authors": ["John Smith"], "topics": ["cloud computing"], 
         "exclude_drafts": true}}

Only include fields that are explicitly mentioned or strongly implied.
"""
        return prompt

    def _simulate_llm_extraction(self, query: str) -> Dict:
        """
        Simulate LLM-based filter extraction.
        In production, replace with actual LLM API call.
        """
        query_lower = query.lower()
        extracted = {}

        # Department detection
        dept_mapping = {
            "technology": "Technology",
            "tech": "Technology",
            "hr": "HR",
            "human resources": "HR",
            "finance": "Finance",
            "financial": "Finance",
            "legal": "Legal",
            "law": "Legal",
            "marketing": "Marketing",
            "engineering": "Engineering",
            "eng": "Engineering",
            "product": "Product",
        }

        found_departments = []
        for keyword, dept in dept_mapping.items():
            if keyword in query_lower:
                found_departments.append(dept)
        if found_departments:
            extracted["departments"] = list(set(found_departments))

        # Exclusion detection
        exclude_patterns = [
            r"exclude\s+(\w+)",
            r"not\s+(?:the\s+)?(\w+)\s+department",
            r"without\s+(\w+)",
            r"except\s+(\w+)",
        ]
        for pattern in exclude_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if match in dept_mapping:
                    if "exclude_departments" not in extracted:
                        extracted["exclude_departments"] = []
                    extracted["exclude_departments"].append(dept_mapping[match])

        # Year detection
        year_pattern = r"\b(20\d{2})\b"
        years = re.findall(year_pattern, query)
        if len(years) == 1:
            extracted["min_year"] = int(years[0])
            extracted["max_year"] = int(years[0])
        elif len(years) >= 2:
            extracted["min_year"] = min(int(y) for y in years)
            extracted["max_year"] = max(int(y) for y in years)

        # Document type detection
        type_mapping = {
            "report": "report",
            "reports": "report",
            "policy": "policy",
            "policies": "policy",
            "memo": "memo",
            "memos": "memo",
            "research paper": "research_paper",
            "research": "research_paper",
            "whitepaper": "whitepaper",
            "white paper": "whitepaper",
            "guide": "guide",
            "guides": "guide",
            "strategy": "strategy",
            "strategies": "strategy",
            "review": "review",
            "reviews": "review",
        }

        found_types = []
        for keyword, dtype in type_mapping.items():
            if keyword in query_lower:
                found_types.append(dtype)
        if found_types:
            extracted["doc_types"] = list(set(found_types))

        # Topic detection
        topics = [
            "ai",
            "machine learning",
            "ml",
            "cloud",
            "security",
            "compliance",
            "data",
            "analytics",
            "blockchain",
            "devops",
            "agile",
        ]
        found_topics = [t for t in topics if t in query_lower]
        if found_topics:
            extracted["topics"] = found_topics

        # Author detection
        author_pattern = r"(?:by|from|written by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        authors = re.findall(author_pattern, query)
        if authors:
            extracted["authors"] = authors

        # Draft exclusion
        if any(
            word in query_lower
            for word in [
                "exclude drafts",
                "no drafts",
                "not draft",
                "published",
                "approved",
            ]
        ):
            if "exclude drafts" in query_lower or "no drafts" in query_lower:
                extracted["exclude_drafts"] = True
            elif "published" in query_lower:
                extracted["status_filter"] = "published"
            elif "approved" in query_lower:
                extracted["status_filter"] = "approved"

        # Credibility detection
        credibility_pattern = r"(?:credibility|quality|reliable)\s*(?:score)?\s*(?:above|over|>=?|at least)\s*(\d+\.?\d*)"
        cred_matches = re.findall(credibility_pattern, query_lower)
        if cred_matches:
            extracted["min_credibility"] = float(cred_matches[0])

        return extracted

    def extract_filters(self, query: str) -> FilterCriteria:
        """Extract filters from a natural language query."""

        # In production: call LLM with prompt
        # response = self.llm.invoke(self._build_extraction_prompt(query))
        # Parse response JSON

        # Simulated extraction
        extracted = self._simulate_llm_extraction(query)

        try:
            return FilterCriteria(**extracted)
        except Exception as e:
            print(f"  Warning: Filter extraction error - {e}")
            return FilterCriteria()

    def extract_and_build_filter(self, query: str) -> Dict:
        """Extract filters and build database-ready filter."""
        criteria = self.extract_filters(query)
        return criteria, self._build_chromadb_filter(criteria)

    def _build_chromadb_filter(self, criteria: FilterCriteria) -> Dict:
        """Convert FilterCriteria to ChromaDB filter format."""
        conditions = []

        if criteria.departments:
            depts = [d.value for d in criteria.departments]
            if len(depts) == 1:
                conditions.append({"department": depts[0]})
            else:
                conditions.append({"department": {"$in": depts}})

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

        if criteria.topics:
            if len(criteria.topics) == 1:
                conditions.append({"topic": criteria.topics[0]})
            else:
                conditions.append({"topic": {"$in": criteria.topics}})

        if criteria.authors:
            if len(criteria.authors) == 1:
                conditions.append({"author": criteria.authors[0]})
            else:
                conditions.append({"author": {"$in": criteria.authors}})

        if criteria.min_credibility is not None:
            conditions.append({"credibility_score": {"$gte": criteria.min_credibility}})

        if criteria.exclude_drafts:
            conditions.append({"status": {"$ne": "draft"}})

        if criteria.status_filter:
            conditions.append({"status": {"$eq": criteria.status_filter}})

        if len(conditions) == 0:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}


def demonstrate_filter_extraction():
    """Demonstrate filter extraction from various queries."""
    print("\n" + "=" * 60)
    print("1. FILTER EXTRACTION FROM QUERIES")
    print("=" * 60)

    extractor = LLMFilterExtractor()

    test_queries = [
        "Show me technology reports from 2023 about AI",
        "What did John Smith write about cloud computing, excluding drafts?",
        "Find published strategies from Finance and Legal in 2024",
        "Show me research papers with credibility above 0.8",
        "What are the HR policies excluding the legal department?",
    ]

    for query in test_queries:
        criteria, db_filter = extractor.extract_and_build_filter(query)

        print(f"\n  Query: '{query}'")
        print(f"  Extracted criteria: {criteria.model_dump(exclude_none=True)}")
        print(f"  Database filter: {db_filter}")

    print("\n✅ Filter extraction complete")


def demonstrate_json_output():
    """Demonstrate JSON serialization of extracted filters."""
    print("\n" + "=" * 60)
    print("2. JSON SERIALIZATION")
    print("=" * 60)

    extractor = LLMFilterExtractor()
    query = (
        "Find technology reports from 2023-2024 about AI and ML, credibility above 0.7"
    )

    criteria, db_filter = extractor.extract_and_build_filter(query)

    # JSON-serializable output
    output = {
        "query": query,
        "extracted_criteria": criteria.model_dump(exclude_none=True),
        "chromadb_filter": db_filter,
        "metadata": {
            "extraction_model": extractor.llm_model,
            "timestamp": datetime.now().isoformat(),
            "criteria_schema": "FilterCriteria v1.0",
        },
    }

    print("\n  Complete extraction output:")
    print(f"  {json.dumps(output, indent=2)}")
    print("\n✅ JSON output ready for API response")


def demonstrate_error_handling():
    """Demonstrate error handling in extraction."""
    print("\n" + "=" * 60)
    print("3. ERROR HANDLING & EDGE CASES")
    print("=" * 60)

    extractor = LLMFilterExtractor()

    edge_cases = [
        ("Empty/no filters", "Tell me about our company"),
        ("Vague query", "What's new?"),
        ("Invalid year", "Show me documents from 1999"),
        ("Conflicting filters", "Show me technology and non-technology reports"),
    ]

    for description, query in edge_cases:
        try:
            criteria, db_filter = extractor.extract_and_build_filter(query)
            print(f"\n  {description}:")
            print(f"    Query: '{query}'")
            print(f"    Criteria: {criteria.model_dump(exclude_none=True)}")
            print(
                f"    Filter: {db_filter if db_filter else '(no filter - broad search)'}"
            )
        except Exception as e:
            print(f"\n  {description}:")
            print(f"    Query: '{query}'")
            print(f"    Error: {e}")
            print("    Recovery: Using empty filter (broad search)")

    print("\n✅ Error handling demonstrated")


def main():
    """Run all custom LLM extractor demonstrations."""
    print("=" * 60)
    print("CUSTOM LLM FILTER EXTRACTOR DEMONSTRATION")
    print("=" * 60)
    print("\nNote: Demonstrates the extraction pipeline.")
    print("Production use requires an actual LLM API.\n")

    demonstrate_filter_extraction()
    demonstrate_json_output()
    demonstrate_error_handling()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Custom extractors give you full control over")
    print("   filter schemas and extraction logic.")


if __name__ == "__main__":
    main()
