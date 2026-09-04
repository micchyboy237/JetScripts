"""
RAG Pitfalls Demo: Retrieval Strategy
Anti-pattern: Top-K Too Small
Solution:     Retrieve wide (top-20) then rerank with cross-encoder to top-5

Shows how retrieving only top-3 misses relevant context, and how a
cross-encoder reranker boosts precision over raw vector scores.

Reference: rag-pitfalls.md → Retrieval Strategy → Top-K Too Small
"""

from jet.adapters.llama_index.factory import (
    get_llama_cpp_embed_model,
    get_llama_cpp_llm,
    get_llama_cpp_reranker,
)
from llama_index.core import Document, Settings, VectorStoreIndex

Settings.llm = get_llama_cpp_llm()
Settings.embed_model = get_llama_cpp_embed_model()

# Larger corpus to make top-K failures visible
documents = [
    Document(
        text="E-4092: Database connection timeout in payment service.",
        metadata={"doc_id": "err-4092"},
    ),
    Document(
        text="E-5001: Invalid API key format in auth gateway.",
        metadata={"doc_id": "err-5001"},
    ),
    Document(
        text="E-3010: Redis cache eviction rate exceeded threshold.",
        metadata={"doc_id": "err-3010"},
    ),
    Document(
        text="E-7721: TLS certificate expired on load balancer.",
        metadata={"doc_id": "err-7721"},
    ),
    Document(
        text="Incident 2024-03-15: Payment service degradation due to AWS us-east-1 outage.",
        metadata={"doc_id": "inc-0315"},
    ),
    Document(
        text="Incident 2024-06-02: Auth gateway latency spike from cert rotation bug.",
        metadata={"doc_id": "inc-0602"},
    ),
    Document(
        text="Best practice: Use connection pooling with 30s idle timeout.",
        metadata={"doc_id": "bp-pooling"},
    ),
    Document(
        text="Best practice: Set Redis maxmemory-policy to allkeys-lru.",
        metadata={"doc_id": "bp-redis"},
    ),
    Document(
        text="SKU-A7X-200: Wireless charging pad, Qi2 compatible.",
        metadata={"doc_id": "sku-a7x"},
    ),
    Document(
        text="SKU-B3K-100: USB-C dock, 100W PD, dual HDMI.",
        metadata={"doc_id": "sku-b3k"},
    ),
    Document(
        text="Runbook: Restart payment-service pods when E-4092 rate > 5/min.",
        metadata={"doc_id": "rb-4092"},
    ),
    Document(
        text="Runbook: Rotate TLS certs 30 days before expiry to avoid E-7721.",
        metadata={"doc_id": "rb-7721"},
    ),
]

vector_index = VectorStoreIndex.from_documents(documents)
query = "How do I fix database connection timeouts in the payment service?"

# ── ❌ ANTI-PATTERN: Top-K Too Small ────────────────────────────────────────
small_retriever = vector_index.as_retriever(similarity_top_k=3)

print("=" * 60)
print("❌ TOP-3 ONLY (may miss critical runbook)")
print("=" * 60)
small_results = small_retriever.retrieve(query)
for i, node in enumerate(small_results):
    print(f"  [{i + 1}] Score: {node.score:.4f} | {node.text[:80]}...")

# ── ✅ SOLUTION: Retrieve Wide + Cross-Encoder Rerank ───────────────────────
wide_retriever = vector_index.as_retriever(similarity_top_k=20)
reranker = get_llama_cpp_reranker(top_n=5)

print("\n" + "=" * 60)
print("✅ RETRIEVE TOP-20 → RERANK TO TOP-5 (via llama.cpp)")
print("=" * 60)
wide_results = wide_retriever.retrieve(query)
reranked_results = reranker.postprocess_nodes(wide_results, query_str=query)
for i, node in enumerate(reranked_results):
    print(f"  [{i + 1}] Score: {node.score:.4f} | {node.text[:80]}...")
