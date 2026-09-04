"""
RAG Pitfalls Demo: Retrieval Strategy
Anti-pattern: No Query Transformation
Solution:     HyDE (Hypothetical Document Embeddings) + Multi-Query

Shows how vague or typo-ridden queries fail with raw embedding, and how
LLM-generated query transformations improve recall.

Reference: rag-pitfalls.md → Retrieval Strategy → No Query Transformation
"""

from jet.adapters.llama_index.factory import (
    get_llama_cpp_embed_model,
    get_llama_cpp_llm,
)
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine

Settings.llm = get_llama_cpp_llm()
Settings.embed_model = get_llama_cpp_embed_model()

documents = [
    Document(
        text="To fix E-4092 database timeout, increase pool size to 20 and set idle_timeout=30s.",
        metadata={"doc_id": "fix-4092"},
    ),
    Document(
        text="Python datetime.utcfromtimestamp() is deprecated; use datetime.fromtimestamp(ts, tz=timezone.utc).",
        metadata={"doc_id": "py-dep"},
    ),
    Document(
        text="Redis OOM: Set maxmemory 2gb and maxmemory-policy allkeys-lru.",
        metadata={"doc_id": "redis-oom"},
    ),
    Document(
        text="Kubernetes CrashLoopBackOff: Check container logs with kubectl logs <pod> --previous.",
        metadata={"doc_id": "k8s-crash"},
    ),
    Document(
        text="AWS S3 AccessDenied: Verify IAM policy includes s3:GetObject for the bucket ARN.",
        metadata={"doc_id": "s3-access"},
    ),
]

vector_index = VectorStoreIndex.from_documents(documents)
base_query_engine = vector_index.as_query_engine(similarity_top_k=3)

# ── ❌ ANTI-PATTERN: Raw Vague Query ────────────────────────────────────────
vague_query = "how do i fix the db thing"

print("=" * 60)
print(f"❌ RAW VAGUE QUERY: '{vague_query}'")
print("=" * 60)
raw_results = base_query_engine.retrieve(vague_query)
for i, node in enumerate(raw_results):
    print(f"  [{i + 1}] Score: {node.score:.4f} | {node.text[:80]}...")

# ── ✅ SOLUTION: HyDE Query Transformation ──────────────────────────────────
hyde_transform = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(
    query_engine=base_query_engine,
    query_transform=hyde_transform,
)

print("\n" + "=" * 60)
print(f"✅ HyDE TRANSFORMED QUERY: '{vague_query}'")
print("=" * 60)
hyde_response = hyde_query_engine.query(vague_query)
print(f"  Answer: {hyde_response.response[:200]}...")
print(f"  Source nodes: {len(hyde_response.source_nodes)}")
for i, node in enumerate(hyde_response.source_nodes):
    print(f"    [{i + 1}] Score: {node.score:.4f} | {node.text[:80]}...")
