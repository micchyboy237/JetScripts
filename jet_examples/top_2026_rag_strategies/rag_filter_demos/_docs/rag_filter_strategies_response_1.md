Response 1
Here are comprehensive code examples for each common RAG filtering case, using popular frameworks like LangChain, LlamaIndex, and direct vector database APIs:

1. Domain-Specific Filtering (Pre-filtering)
   Using ChromaDB with LangChain

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Initialize vector store
vectorstore = Chroma(
    collection_name="company_docs",
    embedding_function=OpenAIEmbeddings(),
)

# INCLUSION FILTER: Search only in Technology department
docs = vectorstore.similarity_search(
    query="What are our AI initiatives?",
    k=5,
    filter={"department": "Technology"}  # Metadata filter
)

# EXCLUSION FILTER: Exclude HR documents
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
```

Using Pinecone

```python
import pinecone

index = pinecone.Index("company-knowledge")

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
```

2. Temporal Filtering (Date Ranges)
   Using Qdrant with Date Filters

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

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
from datetime import datetime

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
```

Using Weaviate with Time Filters

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Filter by date range using GraphQL
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

3. LLM-Powered Intelligent Metadata Extraction
   Using LangChain's Self-Query Retriever

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
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the document was published",
        type="integer",
    ),
    AttributeInfo(
        name="doc_type",
        description="Type of document (report, policy, memo)",
        type="string",
    ),
]

document_content_description = "Company internal documents"
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create self-query retriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True,  # Allow limiting results
)

# User asks natural language question
query = "What were the Technology department's AI initiatives in 2024?"

# The retriever automatically:
# 1. Extracts filters: {"department": "Technology", "year": 2024}
# 2. Applies them before vector search
# 3. Returns relevant docs
docs = retriever.invoke(query)
```

Custom Metadata Extractor with Function Calling

```python
from pydantic import BaseModel, Field
from typing import Optional, List
import json

class SearchFilters(BaseModel):
    """Extracted filters from user query"""
    departments: Optional[List[str]] = Field(
        None,
        description="Departments to include (e.g., ['Technology', 'Finance'])"
    )
    year_min: Optional[int] = Field(
        None,
        description="Minimum year for documents"
    )
    year_max: Optional[int] = Field(
        None,
        description="Maximum year for documents"
    )
    doc_types: Optional[List[str]] = Field(
        None,
        description="Document types to include"
    )
    exclude_departments: Optional[List[str]] = Field(
        None,
        description="Departments to exclude"
    )

def extract_filters_from_query(query: str) -> dict:
    """Use LLM to extract filters from natural language query"""

    prompt = f"""
    Extract search filters from this query: "{query}"

    Return a JSON object matching this schema:
    {{
        "departments": ["Tech"],  // or null
        "year_min": 2023,         // or null
        "year_max": 2024,         // or null
        "doc_types": ["report"],  // or null
        "exclude_departments": ["HR"]  // or null
    }}

    Only include fields that are explicitly mentioned or strongly implied.
    """

    response = llm.invoke(prompt)
    filters = json.loads(response.content)
    return filters

def build_vector_db_filter(extracted_filters: dict) -> dict:
    """Convert extracted filters to vector DB format"""
    conditions = []

    # Handle department inclusion
    if extracted_filters.get("departments"):
        conditions.append({
            "department": {"$in": extracted_filters["departments"]}
        })

    # Handle department exclusion
    if extracted_filters.get("exclude_departments"):
        for dept in extracted_filters["exclude_departments"]:
            conditions.append({
                "department": {"$ne": dept}
            })

    # Handle year range
    if extracted_filters.get("year_min"):
        conditions.append({
            "year": {"$gte": extracted_filters["year_min"]}
        })

    if extracted_filters.get("year_max"):
        conditions.append({
            "year": {"$lte": extracted_filters["year_max"]}
        })

    # Combine with AND
    if len(conditions) == 1:
        return conditions[0]
    elif len(conditions) > 1:
        return {"$and": conditions}
    else:
        return {}  # No filters

# Usage
user_query = "Show me Technology reports from 2023-2024, but exclude HR"
extracted = extract_filters_from_query(user_query)
db_filter = build_vector_db_filter(extracted)

docs = vectorstore.similarity_search(
    query=user_query,
    k=5,
    filter=db_filter
)
```

4. Source Credibility & Document Type Filtering
   Multi-Criteria Filtering Example

```python
from typing import Dict, Any

def search_with_credibility_filters(
    query: str,
    min_credibility_score: float = 0.8,
    allowed_sources: list = None,
    excluded_sources: list = None,
    k: int = 5
) -> list:
    """Search with source credibility filters"""

    filters = []

    # Credibility score filter
    filters.append({
        "credibility_score": {"$gte": min_credibility_score}
    })

    # Source inclusion
    if allowed_sources:
        filters.append({
            "source": {"$in": allowed_sources}
        })

    # Source exclusion
    if excluded_sources:
        for source in excluded_sources:
            filters.append({
                "source": {"$ne": source}
            })

    # Build final filter
    db_filter = {"$and": filters} if len(filters) > 1 else filters[0] if filters else {}

    return vectorstore.similarity_search(
        query=query,
        k=k,
        filter=db_filter
    )

# Usage
results = search_with_credibility_filters(
    query="Climate change impacts",
    min_credibility_score=0.9,
    allowed_sources=["Nature", "Science", "Reuters"],
    excluded_sources=["Blog", "Opinion"]
)
```

Document Type Filtering

```python
# Filter by file type and status
filter_config = {
    "$and": [
        {"file_type": {"$in": ["pdf", "docx"]}},
        {"status": {"$eq": "approved"}},
        {"classification": {"$ne": "confidential"}}
    ]
}

docs = vectorstore.similarity_search(
    query="Project guidelines",
    k=5,
    filter=filter_config
)
```

5. Access Control & Security Filtering
   Row-Level Security with User Context

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_user_permissions(user_id: str) -> dict:
    """Fetch user's access permissions from auth system"""
    # This would call your auth/permission service
    return {
        "allowed_departments": ["Technology", "Product"],
        "security_clearance": "standard",
        "excluded_projects": ["Project_X"]
    }

def secure_search(query: str, user_id: str, k: int = 5) -> list:
    """Search with user-specific access controls"""

    permissions = get_user_permissions(user_id)

    # Build security filter
    security_filter = {
        "$and": [
            # Department access
            {"department": {"$in": permissions["allowed_departments"]}},

            # Security clearance level
            {"clearance_level": {"$lte": permissions["security_clearance"]}},

            # Exclude restricted projects
            {"project": {"$nin": permissions["excluded_projects"]}}
        ]
    }

    return vectorstore.similarity_search(
        query=query,
        k=k,
        filter=security_filter
    )

# Usage
user_docs = secure_search(
    query="Q4 roadmap",
    user_id="user_123"
)
```

Dynamic Filter Based on User Role

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
docs = vectorstore.similarity_search(
    query="Team structure",
    k=5,
    filter=filter_dict
)
```

6. Edge Case Handling with Fallback Strategies

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
        db_filter = build_vector_db_filter(extracted_filters)

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
        extracted_filters={"year_min": 2023},
        fallback_strategy="default_filters"
    )
except ValueError as e:
    print(f"Search failed: {e}")
    # Handle gracefully in UI
```

7. Performance Optimization with Caching

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
    extracted = extract_filters_from_query(query)
    db_filter = build_vector_db_filter(extracted)

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

Complete Production-Ready Example

```python
class IntelligentRAGSearcher:
    """Production-ready RAG searcher with comprehensive filtering"""

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
        """
        Comprehensive search with multiple filter layers

        Filter precedence:
        1. User access controls (security)
        2. Intelligent metadata extraction (if enabled)
        3. Credibility thresholds
        4. Vector similarity
        """

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

These examples cover the most common production scenarios. The key pattern is layered filtering: security → intelligent extraction → quality thresholds → vector search.
