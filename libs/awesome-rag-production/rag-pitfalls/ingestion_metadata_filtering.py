"""
RAG Pitfalls Demo: Data Ingestion & Chunking
Anti-pattern: Ignoring Document Metadata
Solution:     Store metadata + use hybrid filtering for precise retrieval

Shows how metadata enables filtering by category/date/source that pure
semantic search cannot achieve alone.

Reference: rag-pitfalls.md → Data Ingestion → Ignoring Document Metadata
"""

from jet.adapters.llama_index.factory import (
    get_llama_cpp_embed_model,
    get_llama_cpp_llm,
)
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

Settings.llm = get_llama_cpp_llm()
Settings.embed_model = get_llama_cpp_embed_model()

documents = [
    Document(
        text="Payment service v2.3.1 released with connection pooling improvements.",
        metadata={"service": "payment", "version": "2.3.1", "type": "release-note"},
    ),
    Document(
        text="Auth gateway v1.8.0 added OAuth2 PKCE support.",
        metadata={"service": "auth", "version": "1.8.0", "type": "release-note"},
    ),
    Document(
        text="Payment service E-4092 root cause: missing pool_recycle parameter.",
        metadata={"service": "payment", "version": "2.3.0", "type": "postmortem"},
    ),
    Document(
        text="Auth gateway latency spike caused by certificate rotation bug.",
        metadata={"service": "auth", "version": "1.7.9", "type": "postmortem"},
    ),
    Document(
        text="All services should use connection pooling with 30s idle timeout.",
        metadata={"service": "all", "version": "N/A", "type": "best-practice"},
    ),
]

vector_index = VectorStoreIndex.from_documents(documents)
query = "payment service issues"

# ── ❌ ANTI-PATTERN: No Metadata Filtering ──────────────────────────────────
print("=" * 60)
print("❌ NO METADATA FILTER (returns auth docs too)")
print("=" * 60)
unfiltered_results = vector_index.as_retriever(similarity_top_k=5).retrieve(query)
for i, node in enumerate(unfiltered_results):
    svc = node.metadata.get("service", "?")
    typ = node.metadata.get("type", "?")
    print(
        f"  [{i + 1}] Score: {node.score:.4f} | service={svc} type={typ} | {node.text[:60]}..."
    )

# ── ✅ SOLUTION: Metadata Filtering ─────────────────────────────────────────
filters = MetadataFilters(filters=[ExactMatchFilter(key="service", value="payment")])
filtered_retriever = vector_index.as_retriever(
    similarity_top_k=5,
    filters=filters,
)

print("\n" + "=" * 60)
print("✅ WITH METADATA FILTER (service=payment only)")
print("=" * 60)
filtered_results = filtered_retriever.retrieve(query)
for i, node in enumerate(filtered_results):
    svc = node.metadata.get("service", "?")
    typ = node.metadata.get("type", "?")
    print(
        f"  [{i + 1}] Score: {node.score:.4f} | service={svc} type={typ} | {node.text[:60]}..."
    )
