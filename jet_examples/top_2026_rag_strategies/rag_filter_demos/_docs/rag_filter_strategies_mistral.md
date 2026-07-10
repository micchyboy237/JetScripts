# RAG Filter Strategies: The Complete Guide

> **Comprehensive strategies for filtering Retrieval-Augmented Generation (RAG) systems with production-ready code examples**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Domain-Specific Filtering](#1-domain-specific-filtering)
3. [Temporal Filtering](#2-temporal-filtering)
4. [Source Credibility &amp; Document Type Filtering](#3-source-credibility--document-type-filtering)
5. [Access Control &amp; Security Filtering](#4-access-control--security-filtering)
6. [LLM-Powered Intelligent Metadata Extraction](#5-llm-powered-intelligent-metadata-extraction)
7. [Edge Case Handling &amp; Fallback Strategies](#6-edge-case-handling--fallback-strategies)
8. [Performance Optimization](#7-performance-optimization)
9. [Hybrid &amp; Complex Filtering](#8-hybrid--complex-filtering)
10. [Production-Ready Implementations](#9-production-ready-implementations)
11. [Key Takeaways &amp; Best Practices](#10-key-takeaways--best-practices)

---

## Introduction

Retrieval-Augmented Generation (RAG) systems require sophisticated filtering to return relevant, high-quality, and secure documents. This guide provides **production-ready filtering strategies** across multiple dimensions:

- **Domain/Metadata**: Filter by department, topic, author, etc.
- **Temporal**: Filter by date ranges and recency
- **Credibility**: Filter by source quality and trustworthiness
- **Security**: Enforce access controls and compliance rules
- **Intelligent**: Use LLMs to extract filters from natural language queries
- **Hybrid**: Combine multiple filter types with complex logic

Each section includes **code examples** for popular vector databases (ChromaDB, Pinecone, Qdrant, Weaviate, Elasticsearch) and frameworks (LangChain, LlamaIndex).

---

## 1. Domain-Specific Filtering

Filter documents based on metadata fields like department, category, or topic.

### ChromaDB with LangChain

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Initialize vector store
vectorstore = Chroma(
    collection_name="company_docs",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# INCLUSION: Search only in Technology department
docs = vectorstore.similarity_search(
    query="What are our AI initiatives?",
    k=5,
    filter={"department": "Technology"}
)

# EXCLUSION: Exclude HR documents
docs = vectorstore.similarity_search(
    query="Company policies",
    k=5,
    filter={"department": {"$ne": "HR"}}  # $ne = not equal
)

# MULTIPLE INCLUSIONS: Include multiple departments
docs = vectorstore.similarity_search(
    query="Budget approvals",
    k=5,
    filter={"department": {"$in": ["Finance", "Operations"]}}
)

# COMPLEX MULTI-FIELD: Combine multiple metadata conditions
docs = vectorstore.similarity_search(
    query="Recent AI initiatives",
    k=5,
    filter={
        "$and": [
            {"department": "Technology"},
            {"topic": {"$in": ["AI", "Machine Learning"]}},
            {"status": "active"}
        ]
    }
)
```

### Pinecone Implementation

```python
import pinecone

pc = pinecone.Pinecone(api_key="your-api-key")
index = pc.Index("company-knowledge")

# Inclusion filter
query_response = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"department": {"$eq": "Technology"}},
    include_metadata=True
)

# Complex filter with AND logic
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={
        "$and": [
            {"department": {"$in": ["Technology", "Product"]}},
            {"status": {"$eq": "published"}}
        ]
    }
)

# Exclusion using $ne (not equal)
query_response = index.query(
    vector=query_embedding,
    top_k=5,
    filter={
        "department": {"$ne": "Archived"},
        "year": {"$gte": 2023}
    },
    include_metadata=True
)
```

---

## 2. Temporal Filtering

Filter documents by date ranges, recency, or specific time periods.

### Qdrant with Date Filters

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition
from datetime import datetime, timedelta

client = QdrantClient(host="localhost", port=6333)

# INCLUSION: Documents from 2023 onwards
filter_condition = Filter(
    must=[
        FieldCondition(
            key="year",
            range={"gte": 2023}  # Greater than or equal
        )
    ]
)
results = client.search(
    collection_name="news_articles",
    query_vector=query_embedding,
    limit=5,
    query_filter=filter_condition
)

# DATE RANGE: Between specific dates
date_filter = Filter(
    must=[
        FieldCondition(
            key="publish_date",
            range={
                "gte": datetime(2024, 1, 1).timestamp(),
                "lte": datetime(2024, 12, 31).timestamp()
            }
        )
    ]
)

# Include only recent documents (last 12 months)
cutoff_date = (datetime.now() - timedelta(days=365)).timestamp()
search_filter = Filter(
    must=[
        FieldCondition(
            key="timestamp",
            range={"gte": cutoff_date}
        )
    ]
)
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=search_filter,
    limit=10
)
```

### Weaviate with Time Filters

```python
import weaviate

client = weaviate.connect_to_local()

# Filter documents from specific year range
response = client.collections.get("Documents").query.hybrid(
    query="financial performance",
    filters=weaviate.classes.query.Filter.by_property("year").greater_or_equal(2022)
    & weaviate.classes.query.Filter.by_property("year").less_or_equal(2024),
    limit=10
)

# Exclude outdated information (older than 2 years)
current_year = datetime.now().year
min_year = current_year - 2
response = client.collections.get("Documents").query.hybrid(
    query="latest regulations",
    filters=weaviate.classes.query.Filter.by_property("publication_date").greater_than(
        f"{min_year}-01-01T00:00:00Z"
    ),
    limit=10
)

# Using GraphQL syntax
query = """
{
  Get {
    Articles(
      where: {
        operator: And
        operands: [
          {
            path: ["year"]
            operator: GreaterThanEqual
            valueInt: 2023
          },
          {
            path: ["category"]
            operator: Equal
            valueString: "Technology"
          }
        ]
      }
      limit: 5
    ) {
      title
      content
      year
    }
  }
}
"""
result = client.query.raw(query)
```

---

## 3. Source Credibility &amp; Document Type Filtering

Ensure high-quality results by filtering based on source reputation, document type, and verification status.

### Multi-Criteria Source Filtering (Elasticsearch)

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(["http://localhost:9200"])

# Filter by verified sources only
search_query = {
    "size": 10,
    "query": {
        "bool": {
            "must": [
                {
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_embedding,
                        "k": 10,
                        "num_candidates": 100
                    }
                }
            ],
            "filter": [
                {"term": {"source_verified": True}},
                {"terms": {"source_type": ["peer_reviewed", "official", "academic"]}},
                {"range": {"citation_count": {"gte": 10}}}
            ]
        }
    }
}
results = es.search(index="knowledge_base", body=search_query)
```

### Document Type &amp; Status Filtering

```python
# Filter by file format and document type
filter_conditions = {
    "$and": [
        {"file_format": {"$in": ["pdf", "docx"]}},
        {"document_type": {"$in": ["whitepaper", "research_paper", "official_report"]}},
        {"is_draft": {"$eq": False}},  # Exclude drafts
        {"status": {"$eq": "approved"}},
        {"classification": {"$ne": "confidential"}}
    ]
}

results = vectorstore.similarity_search(
    query="market analysis",
    k=10,
    filter=filter_conditions
)

# Credibility score threshold
def search_with_credibility_filters(
    query: str,
    min_credibility_score: float = 0.8,
    allowed_sources: list = None,
    excluded_sources: list = None,
    k: int = 5
) -> list:
    """Search with source credibility filters"""
    filters = []
    filters.append({"credibility_score": {"$gte": min_credibility_score}})

    if allowed_sources:
        filters.append({"source": {"$in": allowed_sources}})

    if excluded_sources:
        for source in excluded_sources:
            filters.append({"source": {"$ne": source}})

    db_filter = {"$and": filters} if len(filters) > 1 else filters[0] if filters else {}
    return vectorstore.similarity_search(query=query, k=k, filter=db_filter)

# Usage
results = search_with_credibility_filters(
    query="Climate change impacts",
    min_credibility_score=0.9,
    allowed_sources=["Nature", "Science", "Reuters"],
    excluded_sources=["Blog", "Opinion"]
)
```

---

## 4. Access Control &amp; Security Filtering

Implement row-level security and compliance filters to ensure users only access authorized documents.

### Row-Level Security with User Context

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_user_permissions(user_id: str) -> dict:
    """Fetch user's access permissions from auth system"""
    return {
        "allowed_departments": ["Technology", "Product"],
        "security_clearance": "standard",
        "excluded_projects": ["Project_X"]
    }

def secure_search(query: str, user_id: str, k: int = 5) -> list:
    """Search with user-specific access controls"""
    permissions = get_user_permissions(user_id)

    security_filter = {
        "$and": [
            {"department": {"$in": permissions["allowed_departments"]}},
            {"clearance_level": {"$lte": permissions["security_clearance"]}},
            {"project": {"$nin": permissions["excluded_projects"]}}
        ]
    }
    return vectorstore.similarity_search(query=query, k=k, filter=security_filter)

# Usage
user_docs = secure_search(query="Q4 roadmap", user_id="user_123")
```

### Class-Based Secure Retriever

```python
class SecureRAGRetriever:
    def __init__(self, vector_store, user_id):
        self.vector_store = vector_store
        self.user_id = user_id
        self.user_permissions = self._get_user_permissions(user_id)

    def _get_user_permissions(self, user_id):
        return {
            "allowed_departments": ["Technology", "Engineering"],
            "security_clearance": "confidential",
            "allowed_projects": ["proj_001", "proj_002"]
        }

    def secure_search(self, query, k=5):
        security_filter = {
            "$and": [
                {"department": {"$in": self.user_permissions["allowed_departments"]}},
                {"security_level": {"$lte": self.user_permissions["security_clearance"]}},
                {"project_id": {"$in": self.user_permissions["allowed_projects"]}}
            ]
        }
        return self.vector_store.similarity_search(query=query, k=k, filter=security_filter)

# Usage
retriever = SecureRAGRetriever(vector_store, user_id="user_123")
results = retriever.secure_search("What's our API strategy?")
```

### Dynamic Filter Based on User Role

```python
def get_role_based_filter(user_role: str) -> dict:
    """Generate filters based on user role"""
    role_filters = {
        "admin": {},  # No restrictions
        "manager": {
            "visibility": {"$in": ["public", "internal", "management"]}
        },
        "employee": {
            "visibility": {"$in": ["public", "internal"]}
        },
        "intern": {
            "visibility": {"$eq": "public"},
            "age_days": {"$lte": 365}  # Only recent docs
        }
    }
    return role_filters.get(user_role, {"visibility": {"$eq": "public"}})

# Usage
filter_dict = get_role_based_filter(user_role="manager")
docs = vectorstore.similarity_search(query="Team structure", k=5, filter=filter_dict)
```

### Dataset Governance with Custom Rules

```python
from typing import Dict

class GovernedRAGSystem:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.governance_rules = self._load_governance_rules()

    def _load_governance_rules(self) -> Dict:
        """Load organization-specific governance rules"""
        return {
            "compliance_filters": {
                "gdpr": {"region": {"$in": ["EU", "EEA"]}},
                "hipaa": {"data_type": {"$ne": "phi"}},  # Exclude PHI
                "sox": {"document_status": "approved"}
            },
            "retention_policies": {
                "max_age_days": 730,  # 2 years
                "archive_after_days": 365
            }
        }

    def compliant_search(self, query: str, compliance_type: str = None, k: int = 5):
        """Search with compliance filters applied"""
        base_filter = {}
        if compliance_type:
            base_filter = self.governance_rules["compliance_filters"].get(compliance_type, {})

        # Add retention policy filter
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(
            days=self.governance_rules["retention_policies"]["max_age_days"]
        )).isoformat()
        retention_filter = {"created_at": {"$gte": cutoff_date}}

        # Combine filters
        final_filter = {
            "$and": [base_filter, retention_filter] if base_filter else retention_filter
        }
        return self.vector_store.similarity_search(query=query, k=k, filter=final_filter)

# Usage
rag_system = GovernedRAGSystem(vector_store)
results = rag_system.compliant_search(query="customer data handling", compliance_type="gdpr")
```

---

## 5. LLM-Powered Intelligent Metadata Extraction

Use LLMs to automatically extract filter criteria from natural language queries.

### LangChain's Self-Query Retriever

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# Define metadata fields that can be filtered
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
        name="doc_type",
        description="Type of document (report, policy, memo)",
        type="string",
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
        name="credibility_score",
        description="Source credibility (0-1)",
        type="float",
    ),
]

document_content_description = "Company internal documents including reports, policies, and technical documentation"
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create self-query retriever
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True
)

# Natural language queries automatically get converted to filtered searches
results = self_query_retriever.invoke(
    "What were the technology department's AI initiatives in 2023?"
)
# Automatically extracts: {"department": "Technology", "year": 2023, "topic": "AI"}
```

### Custom LLM Filter Extractor

````python
from pydantic import BaseModel, Field
from typing import Optional, List
import json

class FilterCriteria(BaseModel):
    """Schema for extracted filter criteria"""
    departments: Optional[List[str]] = Field(
        None,
        description="List of departments to include"
    )
    min_year: Optional[int] = Field(
        None,
        description="Minimum publication year"
    )
    max_year: Optional[int] = Field(
        None,
        description="Maximum publication year"
    )
    topics: Optional[List[str]] = Field(
        None,
        description="List of topics to include"
    )
    authors: Optional[List[str]] = Field(
        None,
        description="List of authors to include"
    )
    exclude_drafts: Optional[bool] = Field(
        False,
        description="Whether to exclude draft documents"
    )
    exclude_departments: Optional[List[str]] = Field(
        None,
        description="Departments to exclude"
    )

def extract_filters_from_query(query: str, llm) -> dict:
    """Extract filter criteria from natural language query"""
    prompt = f"""
    Analyze the following user query and extract filtering criteria.
    Query: "{query}"

    Return ONLY a JSON object with these fields (use null if not specified):
    - departments: array of department names
    - min_year: integer
    - max_year: integer
    - topics: array of topic keywords
    - authors: array of author names
    - exclude_drafts: boolean
    - exclude_departments: array of department names to exclude

    Examples:
    Query: "Show me technology reports from 2023 about AI"
    Output: {{"departments": ["Technology"], "min_year": 2023, "max_year": 2023, "topics": ["AI"]}}

    Query: "What did John Smith write about cloud computing?"
    Output: {{"authors": ["John Smith"], "topics": ["cloud computing"]}}

    Query: "Show me Technology reports from 2023-2024, but exclude HR"
    Output: {{"departments": ["Technology"], "min_year": 2023, "max_year": 2024, "exclude_departments": ["HR"]}}
    """
    response = llm.invoke(prompt)

    try:
        json_str = response.content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3]  # Remove markdown code block
        elif json_str.startswith("```"):
            json_str = json_str[3:-3]
        criteria = FilterCriteria.parse_raw(json_str)
        return build_filter_from_criteria(criteria)
    except Exception as e:
        print(f"Filter extraction failed: {e}")
        return {}  # Return empty filter (no filtering)

def build_filter_from_criteria(criteria: FilterCriteria) -> dict:
    """Convert extracted criteria into vector store filter"""
    conditions = []

    if criteria.departments:
        conditions.append({"department": {"$in": criteria.departments}})
    if criteria.min_year:
        conditions.append({"year": {"$gte": criteria.min_year}})
    if criteria.max_year:
        conditions.append({"year": {"$lte": criteria.max_year}})
    if criteria.topics:
        conditions.append({"topic": {"$in": criteria.topics}})
    if criteria.authors:
        conditions.append({"author": {"$in": criteria.authors}})
    if criteria.exclude_drafts:
        conditions.append({"status": {"$ne": "draft"}})
    if criteria.exclude_departments:
        for dept in criteria.exclude_departments:
            conditions.append({"department": {"$ne": dept}})

    if len(conditions) == 0:
        return {}
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}

# Usage
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
query = "Find recent finance documents from 2024 about budget planning"
filter_dict = extract_filters_from_query(query, llm)
print(f"Extracted filter: {json.dumps(filter_dict, indent=2)}")

results = vectorstore.similarity_search(
    query=query,
    k=10,
    filter=filter_dict if filter_dict else None
)
````

---

## 6. Edge Case Handling &amp; Fallback Strategies

Implement robust error handling and fallback mechanisms for when filter extraction fails.

### Robust Search with Multiple Fallback Strategies

```python
import logging

logger = logging.getLogger(__name__)

def robust_search_with_fallback(
    query: str,
    extracted_filters: dict,
    k: int = 5,
    fallback_strategy: str = "default_filters"
) -> list:
    """
    Search with robust error handling for filter extraction failures

    Strategies:
    - 'default_filters': Apply sensible defaults
    - 'no_filters': Proceed without filters
    - 'strict': Raise error if filters missing
    """
    try:
        # Try to build filter
        db_filter = build_filter_from_criteria(extracted_filters)

        if not db_filter and fallback_strategy == "strict":
            raise ValueError("Required filters could not be extracted")

        # Apply default filters if none extracted
        if not db_filter and fallback_strategy == "default_filters":
            logger.info("No filters extracted, applying defaults")
            db_filter = {
                "year": {"$gte": 2020},  # Default: recent docs only
                "status": {"$eq": "published"}
            }

        # Perform search
        docs = vectorstore.similarity_search(
            query=query,
            k=k,
            filter=db_filter if db_filter else None
        )
        logger.info(f"Search completed with filter: {db_filter}")
        return docs

    except Exception as e:
        logger.error(f"Filter application failed: {e}")
        if fallback_strategy == "no_filters":
            logger.warning("Falling back to unfiltered search")
            return vectorstore.similarity_search(query=query, k=k)
        else:
            raise

# Usage with error handling
try:
    results = robust_search_with_fallback(
        query="What happened in 2023?",
        extracted_filters={"min_year": 2023},
        fallback_strategy="default_filters"
    )
except ValueError as e:
    print(f"Search failed: {e}")
```

### Class-Based Robust Retriever

```python
class RobustRAGRetriever:
    def __init__(self, vector_store, llm, default_filters=None):
        self.vector_store = vector_store
        self.llm = llm
        self.default_filters = default_filters or {"year": {"$gte": 2020}}
        self.query_cache = {}

    def retrieve_with_fallback(self, query: str, k: int = 5, strict_mode: bool = False):
        """Retrieve with multiple fallback strategies"""
        # Strategy 1: Try intelligent filter extraction
        try:
            filter_dict = extract_filters_from_query(query, self.llm)
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
                if len(results) >= 2:
                    return results
                else:
                    print("Warning: Too few results with filters, trying fallback...")
        except Exception as e:
            print(f"Filter extraction failed: {e}")

        # Strategy 2: Apply default filters
        if self.default_filters:
            try:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=self.default_filters
                )
                if len(results) >= 2:
                    return results
            except Exception as e:
                print(f"Default filter search failed: {e}")

        # Strategy 3: No filters (broadest search)
        if not strict_mode:
            results = self.vector_store.similarity_search(
                query=query,
                k=k * 2  # Get more results to compensate
            )
            return results[:k]

        # Strategy 4: Strict mode - refuse to answer
        raise ValueError(
            "Cannot retrieve relevant documents with available filters. "
            "Please refine your query or provide more specific criteria."
        )

    def retrieve_with_cache(self, query: str, k: int = 5):
        """Retrieve with query caching for performance"""
        cache_key = hash(query)
        if cache_key in self.query_cache:
            print("Cache hit!")
            return self.query_cache[cache_key]

        results = self.retrieve_with_fallback(query, k)
        self.query_cache[cache_key] = results

        # Limit cache size
        if len(self.query_cache) > 1000:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        return results

# Usage
robust_retriever = RobustRAGRetriever(
    vector_store=vector_store,
    llm=llm,
    default_filters={"year": {"$gte": 2022}}
)
results = robust_retriever.retrieve_with_fallback(
    "What are our security protocols?",
    strict_mode=False
)
```

---

## 7. Performance Optimization

Improve RAG performance through caching and efficient query strategies.

### Caching Strategies

```python
import hashlib
from cachetools import cached, TTLCache

# Cache for 1 hour
filter_cache = TTLCache(maxsize=1000, ttl=3600)

@cached(cache=filter_cache)
def cached_search(query: str, filter_hash: str, k: int = 5) -> list:
    """Cached search to avoid redundant LLM + DB calls"""
    return vectorstore.similarity_search(query=query, k=k)

def smart_search(query: str, k: int = 5) -> list:
    """Search with intelligent caching"""
    # Extract filters
    extracted = extract_filters_from_query(query, llm)
    db_filter = build_filter_from_criteria(extracted)

    # Create cache key from query + filters
    cache_key = hashlib.md5(
        f"{query}:{json.dumps(db_filter, sort_keys=True)}".encode()
    ).hexdigest()

    # Check cache first
    if cache_key in filter_cache:
        logger.info("Cache hit!")
        return filter_cache[cache_key]

    # Perform search and cache result
    results = vectorstore.similarity_search(
        query=query,
        k=k,
        filter=db_filter if db_filter else None
    )
    filter_cache[cache_key] = results
    return results
```

---

## 8. Hybrid &amp; Complex Filtering

Combine multiple filter types with complex AND/OR logic for sophisticated queries.

### Complex AND/OR Operations

```python
def build_complex_filter():
    """Build complex filter with AND/OR logic"""
    # Example: (Department = Tech OR Department = Engineering)
    # AND (Year >= 2023)
    # AND (Topic = AI OR Topic = ML)
    # AND (Status != draft)
    # AND (Confidence >= 0.7)

    complex_filter = {
        "$and": [
            {
                "$or": [
                    {"department": "Technology"},
                    {"department": "Engineering"}
                ]
            },
            {"year": {"$gte": 2023}},
            {
                "$or": [
                    {"topic": "Artificial Intelligence"},
                    {"topic": "Machine Learning"},
                    {"topic": "AI"},
                    {"topic": "ML"}
                ]
            },
            {"status": {"$ne": "draft"}},
            {"confidence_score": {"$gte": 0.7}}
        ]
    }
    return complex_filter

# Use with vector store
filter_dict = build_complex_filter()
results = vectorstore.similarity_search(
    query="Latest AI developments",
    k=10,
    filter=filter_dict
)
```

### Weaviate Advanced Filtering

```python
import weaviate

# Complex filter with multiple conditions using Weaviate's fluent API
filters = (
    weaviate.classes.query.Filter.by_property("department").equal("Technology")
    | weaviate.classes.query.Filter.by_property("department").equal("Engineering")
) & (
    weaviate.classes.query.Filter.by_property("year").greater_or_equal(2023)
) & (
    weaviate.classes.query.Filter.by_property("topic").like("%AI%")
    | weaviate.classes.query.Filter.by_property("topic").like("%ML%")
)

response = client.collections.get("Documents").query.hybrid(
    query="machine learning advancements",
    filters=filters,
    limit=10
)
```

---

## 9. Production-Ready Implementations

Complete, production-ready classes that combine multiple filtering strategies.

### Intelligent RAG Searcher (Layered Filtering)

```python
from langchain.chains.query_constructor.base import AttributeInfo

class IntelligentRAGSearcher:
    """
    Production-ready RAG searcher with comprehensive filtering.

    Filter precedence:
    1. User access controls (security)
    2. Intelligent metadata extraction (if enabled)
    3. Credibility thresholds
    4. Vector similarity
    """

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.metadata_fields = [
            AttributeInfo(name="department", type="string",
                         description="Document department"),
            AttributeInfo(name="year", type="integer",
                         description="Publication year"),
            AttributeInfo(name="doc_type", type="string",
                         description="Document type"),
            AttributeInfo(name="credibility_score", type="float",
                         description="Source credibility (0-1)"),
        ]

    def search(
        self,
        query: str,
        user_id: str = None,
        min_credibility: float = 0.7,
        k: int = 5,
        use_intelligent_filtering: bool = True
    ) -> list:
        """Comprehensive search with multiple filter layers"""

        # Layer 1: Security filters
        security_filter = self._get_security_filter(user_id) if user_id else {}

        # Layer 2: Intelligent filtering
        if use_intelligent_filtering:
            extracted = self._extract_filters(query)
            intelligent_filter = self._build_filter(extracted)
        else:
            intelligent_filter = {}

        # Layer 3: Credibility filter
        credibility_filter = {"credibility_score": {"$gte": min_credibility}}

        # Combine all filters
        all_filters = self._merge_filters([
            security_filter,
            intelligent_filter,
            credibility_filter
        ])

        # Execute search
        docs = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=all_filters if all_filters else None
        )
        return docs

    def _get_security_filter(self, user_id: str) -> dict:
        """Get user-specific security filters"""
        perms = get_user_permissions(user_id)
        return {
            "$and": [
                {"department": {"$in": perms["allowed_departments"]}},
                {"clearance_level": {"$lte": perms["security_clearance"]}}
            ]
        }

    def _extract_filters(self, query: str) -> dict:
        """Extract filters using LLM"""
        # Implementation from earlier examples
        pass

    def _build_filter(self, extracted: dict) -> dict:
        """Build DB filter from extracted data"""
        # Implementation from earlier examples
        pass

    def _merge_filters(self, filters: list) -> dict:
        """Merge multiple filter dicts with AND logic"""
        valid_filters = [f for f in filters if f]
        if not valid_filters:
            return {}
        elif len(valid_filters) == 1:
            return valid_filters[0]
        else:
            return {"$and": valid_filters}

# Usage
searcher = IntelligentRAGSearcher(vectorstore, llm)
results = searcher.search(
    query="What are our AI plans for next year?",
    user_id="user_456",
    min_credibility=0.85,
    k=5,
    use_intelligent_filtering=True
)
```

### Complete Governed RAG System

```python
class CompleteRAGSystem:
    """
    Complete RAG system with all filtering strategies integrated.
    Combines security, governance, intelligent extraction, and performance optimization.
    """

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.governance_rules = self._load_governance_rules()
        self.filter_cache = TTLCache(maxsize=1000, ttl=3600)

    def _load_governance_rules(self):
        return {
            "compliance_filters": {
                "gdpr": {"region": {"$in": ["EU", "EEA"]}},
                "hipaa": {"data_type": {"$ne": "phi"}},
                "sox": {"document_status": "approved"}
            },
            "retention_policies": {
                "max_age_days": 730,
                "archive_after_days": 365
            }
        }

    def search(
        self,
        query: str,
        user_id: str = None,
        compliance_type: str = None,
        min_credibility: float = 0.8,
        k: int = 5,
        use_intelligent_filtering: bool = True,
        use_cache: bool = True
    ) -> list:
        """Complete search with all filtering layers"""

        # Create cache key
        cache_key = hashlib.md5(
            f"{query}:{user_id}:{compliance_type}:{min_credibility}:{use_intelligent_filtering}".encode()
        ).hexdigest()

        # Check cache
        if use_cache and cache_key in self.filter_cache:
            return self.filter_cache[cache_key]

        # Layer 1: Security filters
        security_filter = {}
        if user_id:
            perms = get_user_permissions(user_id)
            security_filter = {
                "$and": [
                    {"department": {"$in": perms["allowed_departments"]}},
                    {"clearance_level": {"$lte": perms["security_clearance"]}},
                    {"project": {"$nin": perms.get("excluded_projects", [])}}
                ]
            }

        # Layer 2: Compliance filters
        compliance_filter = {}
        if compliance_type:
            compliance_filter = self.governance_rules["compliance_filters"].get(
                compliance_type, {}
            )

        # Layer 3: Retention policy
        cutoff_date = (datetime.now() - timedelta(
            days=self.governance_rules["retention_policies"]["max_age_days"]
        )).isoformat()
        retention_filter = {"created_at": {"$gte": cutoff_date}}

        # Layer 4: Intelligent filtering
        intelligent_filter = {}
        if use_intelligent_filtering:
            extracted = extract_filters_from_query(query, self.llm)
            intelligent_filter = build_filter_from_criteria(extracted)

        # Layer 5: Credibility filter
        credibility_filter = {"credibility_score": {"$gte": min_credibility}}

        # Combine all filters
        all_filters = self._merge_filters([
            security_filter,
            compliance_filter,
            retention_filter,
            intelligent_filter,
            credibility_filter
        ])

        # Execute search
        docs = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=all_filters if all_filters else None
        )

        # Cache results
        if use_cache:
            self.filter_cache[cache_key] = docs

        return docs

    def _merge_filters(self, filters: list) -> dict:
        """Merge multiple filter dicts with AND logic"""
        valid_filters = [f for f in filters if f]
        if not valid_filters:
            return {}
        elif len(valid_filters) == 1:
            return valid_filters[0]
        else:
            return {"$and": valid_filters}

# Usage
rag_system = CompleteRAGSystem(vectorstore, llm)
results = rag_system.search(
    query="What are our GDPR-compliant AI initiatives in Europe?",
    user_id="user_123",
    compliance_type="gdpr",
    min_credibility=0.85,
    k=5
)
```

---

## 10. Key Takeaways &amp; Best Practices

### 🎯 Filtering Strategy Selection Guide

| Use Case                  | Recommended Strategy   | Complexity | Performance |
| ------------------------- | ---------------------- | ---------- | ----------- |
| Simple metadata filtering | Basic metadata filters | Low        | High        |
| Date-based queries        | Temporal filtering     | Low        | High        |
| User-specific access      | Row-level security     | Medium     | High        |
| Natural language queries  | LLM-powered extraction | High       | Medium      |
| High-reliability needs    | Fallback strategies    | Medium     | Medium      |
| Production systems        | Layered filtering      | High       | Medium      |
| Complex business rules    | Hybrid filtering       | High       | Low         |

### 🏆 Best Practices

1. **Layer Your Filters**: Apply filters in order of importance:

- Security → Compliance → Intelligent → Quality → Similarity

2. **Start Simple**: Begin with basic metadata filtering, then add complexity as needed.
3. **Cache Aggressively**: Cache both filter extraction results and search results.
4. **Handle Edge Cases**: Always implement fallback strategies for filter extraction failures.
5. **Monitor Performance**: Track query latency and result quality with different filter combinations.
6. **Validate Filters**: Ensure extracted filters make sense before applying them.
7. **Document Your Schema**: Clearly document all metadata fields available for filtering.
8. **Test Thoroughly**: Test with various query types and edge cases.

### 📊 Filter Performance Comparison

| Filter Type    | Query Speed | Result Quality | Implementation Complexity |
| -------------- | ----------- | -------------- | ------------------------- |
| Metadata       | ⚡ Fastest  | ✅ Good        | ⭐ Low                    |
| Temporal       | ⚡ Fast     | ✅ Good        | ⭐ Low                    |
| Credibility    | ⚡ Fast     | ✅✅ Excellent | ⭐⭐ Medium               |
| Security       | ⚡ Fast     | ✅✅ Excellent | ⭐⭐ Medium               |
| LLM Extraction | 🐢 Slow     | ✅✅✅ Best    | ⭐⭐⭐ High               |
| Hybrid         | 🐢 Slowest  | ✅✅✅ Best    | ⭐⭐⭐⭐ Very High        |

### 🔧 Recommended Tool Stack

| Component  | Recommended Options                        |
| ---------- | ------------------------------------------ |
| Vector DB  | ChromaDB, Pinecone, Qdrant, Weaviate       |
| Framework  | LangChain, LlamaIndex                      |
| LLM        | GPT-4, Claude 3, Llama 3                   |
| Embeddings | OpenAI, HuggingFace, Sentence Transformers |
| Caching    | cachetools, Redis                          |

---

## Quick Reference: Filter Syntax by Database

### ChromaDB

```python
# Equality
{"department": "Technology"}

# Not equal
{"department": {"$ne": "HR"}}

# In list
{"department": {"$in": ["Tech", "Engineering"]}}

# AND
{"$and": [{"dept": "Tech"}, {"year": {"$gte": 2023}}]}

# OR
{"$or": [{"dept": "Tech"}, {"dept": "Engineering"}]}
```

### Pinecone

```python
# Equality
{"department": {"$eq": "Technology"}}

# Not equal
{"department": {"$ne": "Archived"}}

# Greater than
{"year": {"$gte": 2023}}

# AND
{"$and": [{"dept": {"$eq": "Tech"}}, {"year": {"$gte": 2023}}]}
```

### Qdrant

```python
from qdrant_client.http.models import Filter, FieldCondition

# Greater than
Filter(must=[FieldCondition(key="year", range={"gte": 2023})])

# In list
Filter(must=[FieldCondition(key="dept", match={"value": "Tech"})])
```

### Weaviate

```python
# Greater than
Filter.by_property("year").greater_or_equal(2023)

# OR
Filter.by_property("dept").equal("Tech") | Filter.by_property("dept").equal("Eng")

# AND
Filter.by_property("year").greater_or_equal(2023) & Filter.by_property("status").equal("active")
```

---

_Last updated: July 9, 2026_  
_Maintainer: Lebron James Harden_
