from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# SurrealDB

[SurrealDB](https://surrealdb.com) is a unified, multi-model database purpose-built for AI systems. It combines structured and unstructured data (including vector search, graph traversal, relational queries, full-text search, document storage, and time-series data) into a single ACID-compliant engine, scaling from a 3 MB edge binary to petabyte-scale clusters in the cloud. By eliminating the need for multiple specialized stores, SurrealDB simplifies architectures, reduces latency, and ensures consistency for AI workloads.

**Why SurrealDB Matters for GenAI Systems**
- **One engine for storage and memory:** Combine durable storage and fast, agent-friendly memory in a single system, providing all the data your agent needs and removing the need to sync multiple systems.
- **One-hop memory for agents:** Run vector search, graph traversal, semantic joins, and transactional writes in a single query, giving LLM agents fast, consistent memory access without stitching relational, graph and vector databases together.
- **In-place inference and real-time updates:** SurrealDB enables agents to run inference next to data and receive millisecond-fresh updates, critical for real-time reasoning and collaboration.
- **Versioned, durable context:** SurrealDB supports time-travel queries and versioned records, letting agents audit or “replay” past states for consistent, explainable reasoning.
- **Plug-and-play agent memory:** Expose AI memory as a native concept, making it easy to use SurrealDB as a drop-in backend for AI frameworks.

## Installation and Setup
"""
logger.info("# SurrealDB")

pip install langchain-surrealdb

"""
## Vector Store

[This notebook](/docs/integrations/vectorstores/surrealdb) covers how to get started with the SurrealDB vector store.

Find more [examples](https://github.com/surrealdb/langchain-surrealdb/blob/main/README.md#simple-example) in the repository.
"""
logger.info("## Vector Store")

logger.info("\n\n[DONE]", bright=True)