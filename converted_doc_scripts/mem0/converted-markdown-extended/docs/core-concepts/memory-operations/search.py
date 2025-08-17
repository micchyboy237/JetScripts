import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import Memory
from mem0 import MemoryClient
import os
import shutil
import { Memory } from 'mem0ai/oss'
import { MemoryClient } from "mem0ai"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Search Memory
description: Retrieve relevant memories from Mem0 using powerful semantic and filtered search capabilities.
icon: "magnifying-glass"
iconType: "solid"
---

## Overview

The `search` operation allows you to retrieve relevant memories based on a natural language query and optional filters like user ID, agent ID, categories, and more. This is the foundation of giving your agents memory-aware behavior.

Mem0 supports:
- Semantic similarity search
- Metadata filtering (with advanced logic)
- Reranking and thresholds
- Cross-agent, multi-session context resolution

This applies to both:
- **Mem0 Platform** (hosted API with full-scale features)
- **Mem0 Open Source** (local-first with LLM inference and local vector DB)


## Architecture

<Frame caption="Architecture diagram illustrating the memory search process.">
  <img src="../../images/search_architecture.png" />
</Frame>

The search flow follows these steps:

1. **Query Processing**
   An LLM refines and optimizes your natural language query.

2. **Vector Search**
   Semantic embeddings are used to find the most relevant memories using cosine similarity.

3. **Filtering & Ranking**
   Logical and comparison-based filters are applied. Memories are scored, filtered, and optionally reranked.

4. **Results Delivery**
   Relevant memories are returned with associated metadata and timestamps.

---

## Example: Mem0 Platform

<CodeGroup>
"""
logger.info("## Overview")


client = MemoryClient(api_key="your-api-key")

query = "What do you know about me?"
filters = {
   "OR": [
      {"user_id": "alice"},
      {"agent_id": {"in": ["travel-assistant", "customer-support"]}}
   ]
}

results = client.search(query, version="v2", filters=filters)

"""

"""


client = new MemoryClient({apiKey: "your-api-key"})

query = "I'm craving some pizza. Any recommendations?"
filters = {
  AND: [
    { user_id: "alice" }
  ]
}

async def run_async_code_5b68c58f():
    results = await client.search(query, {
    return results
results = asyncio.run(run_async_code_5b68c58f())
logger.success(format_json(results))
  version: "v2",
  filters
})

"""
</CodeGroup>

---

## Example: Mem0 Open Source

<CodeGroup>
"""
logger.info("## Example: Mem0 Open Source")


m = Memory()
related_memories = m.search("Should I drink coffee or tea?", user_id="alice")

"""

"""


memory = new Memory()
relatedMemories = memory.search("Should I drink coffee or tea?", { userId: "alice" })

"""
</CodeGroup>

---

## Tips for Better Search

- Use descriptive natural queries (Mem0 can interpret intent)
- Apply filters for scoped, faster lookup
- Use `version: "v2"` for enhanced results
- Consider wildcard filters (e.g., `run_id: "*"`) for broader matches
- Tune with `top_k`, `threshold`, or `rerank` if needed


### More Details

For the full list of filter logic, comparison operators, and optional search parameters, see the
[Search Memory API Reference](/api-reference/memory/v2-search-memories).

---

## Need help?
If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx"/>
"""
logger.info("## Tips for Better Search")

logger.info("\n\n[DONE]", bright=True)