"""
RAG Pitfalls Demo: Production Deployment
Anti-pattern: Synchronous Retrieval in API
Solution:     Async retrieval + streaming LLM response

Demonstrates non-blocking async retrieval and token-by-token streaming
to reduce time-to-first-token in production APIs.

Reference: rag-pitfalls.md → Production → Synchronous Retrieval in API
"""

import asyncio
import time

from jet.adapters.llama_index.factory import (
    get_llama_cpp_embed_model,
    get_llama_cpp_llm,
)
from llama_index.core import Document, Settings, VectorStoreIndex

Settings.llm = get_llama_cpp_llm()
Settings.embed_model = get_llama_cpp_embed_model()

documents = [
    Document(text="E-4092: Database connection timeout. Increase pool_size to 20."),
    Document(text="E-5001: Invalid API key. Must start with jk_live_ or jk_test_."),
    Document(text="Best practice: Use connection pooling with 30s idle timeout."),
]

vector_index = VectorStoreIndex.from_documents(documents)
query = "How do I fix database connection timeouts?"


# ── ❌ ANTI-PATTERN: Synchronous Blocking ───────────────────────────────────
def sync_query():
    engine = vector_index.as_query_engine(similarity_top_k=3, streaming=False)
    return engine.query(query)


# ── ✅ SOLUTION: Async + Streaming ──────────────────────────────────────────
async def async_streaming_query():
    engine = vector_index.as_query_engine(similarity_top_k=3, streaming=True)
    response = await engine.aquery(query)

    # Collect streamed tokens
    tokens = []
    async for token in response.async_response_gen():
        tokens.append(token)
    return "".join(tokens)


async def main():
    # Sync benchmark
    t0 = time.perf_counter()
    sync_result = sync_query()
    sync_time = time.perf_counter() - t0

    print("=" * 60)
    print("❌ SYNCHRONOUS (blocking)")
    print("=" * 60)
    print(f"  Time: {sync_time:.3f}s")
    print(f"  Response: {str(sync_result)[:200]}...")

    # Async streaming benchmark
    t0 = time.perf_counter()
    stream_result = await async_streaming_query()
    stream_time = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print("✅ ASYNC + STREAMING")
    print("=" * 60)
    print(f"  Time: {stream_time:.3f}s")
    print(f"  Response: {stream_result[:200]}...")
    print(f"\n  ⚡ Speedup: {sync_time / stream_time:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
