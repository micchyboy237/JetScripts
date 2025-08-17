from jet.logger import CustomLogger
from mem0 import MemoryClient
import MemoryClient from 'mem0ai'
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Delete Memory
description: Remove memories from Mem0 either individually, in bulk, or via filters.
icon: "trash"
iconType: "solid"
---

## Overview

Memories can become outdated, irrelevant, or need to be removed for privacy or compliance reasons. Mem0 offers flexible ways to delete memory:

1. **Delete a Single Memory**: Using a specific memory ID
2. **Batch Delete**: Delete multiple known memory IDs (up to 1000)
3. **Filtered Delete**: Delete memories matching a filter (e.g., `user_id`, `metadata`, `run_id`)

This page walks through code example for each method.


## Use Cases

- Forget a user’s past preferences by request
- Remove outdated or incorrect memory entries
- Clean up memory after session expiration
- Comply with data deletion requests (e.g., GDPR)

---

## 1. Delete a Single Memory by ID

<CodeGroup>
"""
logger.info("## Overview")


client = MemoryClient(api_key="your-api-key")

memory_id = "your_memory_id"
client.delete(memory_id=memory_id)

"""

"""


client = new MemoryClient({ apiKey: "your-api-key" })

client.delete("your_memory_id")
  .then(result => console.log(result))
  .catch(error => console.error(error))

"""
</CodeGroup>

---

## 2. Batch Delete Multiple Memories

<CodeGroup>
"""
logger.info("## 2. Batch Delete Multiple Memories")


client = MemoryClient(api_key="your-api-key")

delete_memories = [
    {"memory_id": "id1"},
    {"memory_id": "id2"}
]

response = client.batch_delete(delete_memories)
logger.debug(response)

"""

"""


client = new MemoryClient({ apiKey: "your-api-key" })

deleteMemories = [
  { memory_id: "id1" },
  { memory_id: "id2" }
]

client.batchDelete(deleteMemories)
  .then(response => console.log('Batch delete response:', response))
  .catch(error => console.error(error))

"""
</CodeGroup>

---

## 3. Delete Memories by Filter (e.g., user_id)

<CodeGroup>
"""
logger.info("## 3. Delete Memories by Filter (e.g., user_id)")


client = MemoryClient(api_key="your-api-key")

client.delete_all(user_id="alice")

"""

"""


client = new MemoryClient({ apiKey: "your-api-key" })

client.deleteAll({ user_id: "alice" })
  .then(result => console.log(result))
  .catch(error => console.error(error))

"""
</CodeGroup>

You can also filter by other parameters such as:
- `agent_id`
- `run_id`
- `metadata` (as JSON string)

---

## Key Differences

| Method                | Use When                                | IDs Needed | Filters |
|----------------------|-------------------------------------------|------------|----------|
| `delete(memory_id)`  | You know exactly which memory to remove   | ✔          | ✘        |
| `batch_delete([...])`| You have a known list of memory IDs       | ✔          | ✘        |
| `delete_all(...)`    | You want to delete by user/agent/run/etc | ✘          | ✔        |


### More Details

For request/response schema and additional filtering options, see:
- [Delete Memory API Reference](/api-reference/memory/delete-memory)
- [Batch Delete API Reference](/api-reference/memory/batch-delete)
- [Delete Memories by Filter Reference](/api-reference/memory/delete-memories)

You’ve now seen how to add, search, update, and delete memories in Mem0.

---

## Need help?
If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx"/>
"""
logger.info("## Key Differences")

logger.info("\n\n[DONE]", bright=True)