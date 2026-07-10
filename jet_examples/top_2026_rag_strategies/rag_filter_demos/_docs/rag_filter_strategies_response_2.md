Response 2
Here are comprehensive code examples for each RAG filtering case:

1. Domain-Specific Filtering
   Basic Metadata Filtering (ChromaDB)

```python
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Initialize vector store with metadata
embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    collection_name="company_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# INCLUSION FILTER: Search only in Technology department
results = vector_store.similarity_search(
    query="What are our cloud migration strategies?",
    k=5,
    filter={"department": "Technology"}
)

# EXCLUSION via multiple inclusion: Include HR and Finance, exclude others
results = vector_store.similarity_search(
    query="Show me policy updates",
    k=5,
    filter={"department": {"$in": ["HR", "Finance"]}}
)

# Complex multi-field filtering
results = vector_store.similarity_search(
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

Pinecone Implementation

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index("company-knowledge")

# Inclusion filter
query_response = index.query(
    vector=query_embedding,
    top_k=5,
    filter={
        "department": {"$eq": "Technology"}
    },
    include_metadata=True
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

2. Temporal Filtering
   Date Range Filtering

```python
from datetime import datetime
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
```

Qdrant Date Filtering

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

client = QdrantClient(url="http://localhost:6333")

# Include only recent documents (last 12 months)
from datetime import datetime, timedelta

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
    search_params={"exact": False},
    query_filter=search_filter,
    limit=10
)
```

3. Source Credibility Filtering
   Multi-Criteria Source Filtering

```python
# Elasticsearch + Vector Search
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

Document Type Filtering

```python
# Filter by file format and document type
filter_conditions = {
    "$and": [
        {"file_format": {"$in": ["pdf", "docx"]}},
        {"document_type": {"$in": ["whitepaper", "research_paper", "official_report"]}},
        {"is_draft": {"$eq": False}}  # Exclude drafts
    ]
}

results = vector_store.similarity_search(
    query="market analysis",
    k=10,
    filter=filter_conditions
)
```

4. Access Control & Security
   Row-Level Security with User Permissions

```python
class SecureRAGRetriever:
    def __init__(self, vector_store, user_id):
        self.vector_store = vector_store
        self.user_id = user_id
        self.user_permissions = self._get_user_permissions(user_id)

    def _get_user_permissions(self, user_id):
        """Fetch user's access permissions from auth system"""
        # Example: Query your auth database
        return {
            "allowed_departments": ["Technology", "Engineering"],
            "security_clearance": "confidential",
            "allowed_projects": ["proj_001", "proj_002"]
        }

    def secure_search(self, query, k=5):
        """Perform filtered search based on user permissions"""

        # Build security filter
        security_filter = {
            "$and": [
                {"department": {"$in": self.user_permissions["allowed_departments"]}},
                {"security_level": {"$lte": self.user_permissions["security_clearance"]}},
                {"project_id": {"$in": self.user_permissions["allowed_projects"]}}
            ]
        }

        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=security_filter
        )

        return results

# Usage
retriever = SecureRAGRetriever(vector_store, user_id="user_123")
results = retriever.secure_search("What's our API strategy?")
```

Dataset Governance with Custom Rules

```python
from typing import List, Dict

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

        retention_filter = {
            "created_at": {"$gte": cutoff_date}
        }

        # Combine filters
        final_filter = {
            "$and": [base_filter, retention_filter] if base_filter else retention_filter
        }

        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=final_filter
        )

# Usage
rag_system = GovernedRAGSystem(vector_store)
results = rag_system.compliant_search(
    query="customer data handling",
    compliance_type="gdpr"
)
```

5. Intelligent Metadata Extraction (LLM-Powered)
   Using LangChain's SelfQueryRetriever

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

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
        name="topic",
        description="The main topic of the document",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="The author of the document",
        type="string",
    ),
]

document_content_description = "Company internal documents including reports, policies, and technical documentation"

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create self-query retriever
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_store,
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

Custom LLM Filter Extractor

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

    Examples:
    Query: "Show me technology reports from 2023 about AI"
    Output: {{"departments": ["Technology"], "min_year": 2023, "max_year": 2023, "topics": ["AI"]}}

    Query: "What did John Smith write about cloud computing?"
    Output: {{"authors": ["John Smith"], "topics": ["cloud computing"]}}
    """

    response = llm.invoke(prompt)

    try:
        # Parse JSON from response
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

results = vector_store.similarity_search(
    query=query,
    k=10,
    filter=filter_dict if filter_dict else None
)
````

6. Edge Case Handling
   Fallback Strategies

```python
class RobustRAGRetriever:
    def __init__(self, vector_store, llm, default_filters=None):
        self.vector_store = vector_store
        self.llm = llm
        self.default_filters = default_filters or {
            "year": {"$gte": 2020}  # Default: last 4 years
        }
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

                # Validate results quality
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
                k=k * 2  # Get more results to compensate for lack of filtering
            )
            return results[:k]

        # Strategy 4: Strict mode - refuse to answer
        raise ValueError(
            "Cannot retrieve relevant documents with available filters. "
            "Please refine your query or provide more specific criteria."
        )

    def retrieve_with_cache(self, query: str, k: int = 5):
        """Retrieve with query caching for performance"""

        # Check cache
        cache_key = hash(query)
        if cache_key in self.query_cache:
            print("Cache hit!")
            return self.query_cache[cache_key]

        # Perform retrieval
        results = self.retrieve_with_fallback(query, k)

        # Cache results
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

# This will try multiple strategies
results = robust_retriever.retrieve_with_fallback(
    "What are our security protocols?",
    strict_mode=False
)
```

7. Hybrid Filtering (Multiple Criteria)
   Complex AND/OR Operations

```python
def build_complex_filter():
    """Build complex filter with AND/OR logic"""

    # Example: (Department = Tech OR Department = Engineering)
    # AND (Year >= 2023)
    # AND (Topic = AI OR Topic = ML)
    # AND (Status != draft)

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
            {"confidence_score": {"$gte": 0.7}}  # Quality threshold
        ]
    }

    return complex_filter

# Use with vector store
filter_dict = build_complex_filter()
results = vector_store.similarity_search(
    query="Latest AI developments",
    k=10,
    filter=filter_dict
)
```

Weaviate Advanced Filtering

```python
import weaviate

# Complex filter with multiple conditions
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

These examples cover the most common RAG filtering scenarios with production-ready implementations. The key is choosing the right approach based on your use case: simple metadata filters for straightforward needs, LLM-powered extraction for user-friendly interfaces, and robust fallback strategies for reliability.
