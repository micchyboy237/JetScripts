"""
RAG Pitfalls Demo: Retrieval Strategy
Anti-pattern: Pure Vector Search Only
Solution:     Hybrid Search (Dense + BM25 + RRF)

Demonstrates why relying solely on semantic similarity fails for exact-match
queries (error codes, product IDs) and how hybrid retrieval fixes it.

Reference: rag-pitfalls.md → Retrieval Strategy → Pure Vector Search Only
"""

from jet.adapters.llama_index.factory import (
    get_llama_cpp_embed_model,
    get_llama_cpp_llm,
)
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

Settings.llm = get_llama_cpp_llm()
Settings.embed_model = get_llama_cpp_embed_model()

documents = [
    Document(
        text="Error code E-4092 indicates a database connection timeout in the payment service.",
        metadata={"doc_id": "err-4092", "category": "error"},
    ),
    Document(
        text="Error code E-5001 indicates an invalid API key format in the auth gateway.",
        metadata={"doc_id": "err-5001", "category": "error"},
    ),
    Document(
        text="The payment service experienced degraded performance on 2024-03-15 due to AWS us-east-1 outage.",
        metadata={"doc_id": "incident-0315", "category": "incident"},
    ),
    Document(
        text="Product SKU-A7X-200 is a wireless charging pad compatible with Qi2 standard.",
        metadata={"doc_id": "sku-a7x", "category": "product"},
    ),
    Document(
        text="Database connection best practices include using connection pooling and setting idle timeouts to 30s.",
        metadata={"doc_id": "db-best-practices", "category": "best-practice"},
    ),
]

vector_index = VectorStoreIndex.from_documents(documents)
query = "What does error code E-4092 mean?"

# ── ❌ ANTI-PATTERN: Pure Vector Search Only ────────────────────────────────
pure_vector_retriever = vector_index.as_retriever(similarity_top_k=3)

print("=" * 60)
print("❌ PURE VECTOR SEARCH RESULTS")
print("=" * 60)
vector_results = pure_vector_retriever.retrieve(query)
for i, node in enumerate(vector_results):
    print(f"  [{i + 1}] Score: {node.score:.4f} | {node.text[:80]}...")

# ── ✅ SOLUTION: Hybrid Search (Dense + BM25 + RRF) ────────────────────────
dense_retriever = vector_index.as_retriever(similarity_top_k=5)
sparse_retriever = BM25Retriever.from_defaults(
    nodes=vector_index.docstore.docs.values(),
    similarity_top_k=5,
)
hybrid_retriever = QueryFusionRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    similarity_top_k=5,
    num_queries=1,
    mode="reciprocal_rerank",
    verbose=True,
)

print("\n" + "=" * 60)
print("✅ HYBRID SEARCH RESULTS (Dense + BM25 + RRF)")
print("=" * 60)
hybrid_results = hybrid_retriever.retrieve(query)
for i, node in enumerate(hybrid_results):
    print(f"  [{i + 1}] Score: {node.score:.4f} | {node.text[:80]}...")
