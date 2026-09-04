"""
RAG Pitfalls Demo: Data Ingestion & Chunking
Anti-pattern: Fixed Chunk Size Everywhere
Solution:     Semantic chunking + document-type-aware sizing + sliding window

Compares naive fixed-size chunking vs. semantic chunking on mixed-content
documents (technical docs with code blocks and tables).

Reference: rag-pitfalls.md → Data Ingestion → Fixed Chunk Size Everywhere
"""

from jet.adapters.llama_index.factory import (
    get_llama_cpp_embed_model,
    get_llama_cpp_llm,
)
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter

Settings.llm = get_llama_cpp_llm()
Settings.embed_model = get_llama_cpp_embed_model()

# Document with mixed content types that break fixed-size chunking
mixed_document = Document(
    text="""# Payment Service Configuration Guide

## Connection Pool Settings
The payment service uses SQLAlchemy connection pooling. Configure as follows:

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
)
```

## Error Code Reference
| Code   | Description                          | Severity |
|--------|--------------------------------------|----------|
| E-4092 | Database connection timeout          | HIGH     |
| E-4093 | Connection pool exhausted            | CRITICAL |
| E-5001 | Invalid API key format               | MEDIUM   |

## Troubleshooting
If you encounter E-4092 errors:
1. Check database server health
2. Verify pool_size is not exceeded
3. Review network latency between app and DB
4. Consider increasing pool_timeout from default 30s""",
    metadata={"doc_type": "technical_guide", "service": "payment"},
)

query = "What is the correct pool_size configuration for the payment service?"

# ── ❌ ANTI-PATTERN: Fixed Chunk Size ───────────────────────────────────────
fixed_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=0)
fixed_nodes = fixed_splitter.get_nodes_from_documents([mixed_document])
fixed_index = VectorStoreIndex(nodes=fixed_nodes)

print("=" * 60)
print(f"❌ FIXED CHUNK SIZE (128 tokens, no overlap)")
print(f"   Chunks created: {len(fixed_nodes)}")
print("=" * 60)
fixed_results = fixed_index.as_retriever(similarity_top_k=3).retrieve(query)
for i, node in enumerate(fixed_results):
    print(f"  [{i + 1}] Score: {node.score:.4f} | {node.text[:80]}...")

# ── ✅ SOLUTION: Semantic Chunking ──────────────────────────────────────────
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=Settings.embed_model,
)
semantic_nodes = semantic_splitter.get_nodes_from_documents([mixed_document])
semantic_index = VectorStoreIndex(nodes=semantic_nodes)

print("\n" + "=" * 60)
print(f"✅ SEMANTIC CHUNKING (breakpoint at 95th percentile)")
print(f"   Chunks created: {len(semantic_nodes)}")
print("=" * 60)
semantic_results = semantic_index.as_retriever(similarity_top_k=3).retrieve(query)
for i, node in enumerate(semantic_results):
    print(f"  [{i + 1}] Score: {node.score:.4f} | {node.text[:80]}...")
