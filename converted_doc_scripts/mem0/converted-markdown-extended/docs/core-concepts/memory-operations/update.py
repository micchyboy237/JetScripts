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
title: Update Memory
description: Modify an existing memory by updating its content or metadata.
icon: "pencil"
iconType: "solid"
---

## Overview

User preferences, interests, and behaviors often evolve over time. The `update` operation lets you revise a stored memory, whether it's updating facts and memories, rephrasing a message, or enriching metadata.

Mem0 supports both:
- **Single Memory Update** for one specific memory using its ID
- **Batch Update** for updating many memories at once (up to 1000)

This guide includes usage for both single update and batch update of memories through **Mem0 Platform**


## Use Cases

- Refine a vague or incorrect memory after a correction
- Add or edit memory with new metadata (e.g., categories, tags)
- Evolve factual knowledge as the user’s profile changes
- A user profile evolves: “I love spicy food” → later says “Actually, I can’t handle spicy food.”

Updating memory ensures your agents remain accurate, adaptive, and personalized.

---

## Update Memory

<CodeGroup>
"""
logger.info("## Overview")


client = MemoryClient(api_key="your-api-key")

memory_id = "your_memory_id"
client.update(
    memory_id=memory_id,
    text="Updated memory content about the user",
    metadata={"category": "profile-update"}
)

"""

"""


client = new MemoryClient({ apiKey: "your-api-key" })
memory_id = "your_memory_id"

client.update(memory_id, {
  text: "Updated memory content about the user",
  metadata: { category: "profile-update" }
})
  .then(result => console.log(result))
  .catch(error => console.error(error))

"""
</CodeGroup>

---

## Batch Update

Update up to 1000 memories in one call.

<CodeGroup>
"""
logger.info("## Batch Update")


client = MemoryClient(api_key="your-api-key")

update_memories = [
    {"memory_id": "id1", "text": "Watches football"},
    {"memory_id": "id2", "text": "Likes to travel"}
]

response = client.batch_update(update_memories)
logger.debug(response)

"""

"""


client = new MemoryClient({ apiKey: "your-api-key" })

updateMemories = [
  { memoryId: "id1", text: "Watches football" },
  { memoryId: "id2", text: "Likes to travel" }
]

client.batchUpdate(updateMemories)
  .then(response => console.log('Batch update response:', response))
  .catch(error => console.error(error))

"""
</CodeGroup>

---

## Tips

- You can update both `text` and `metadata` in the same call.
- Use `batchUpdate` when you're applying similar corrections at scale.
- If memory is marked `immutable`, it must first be deleted and re-added.
- Combine this with feedback mechanisms (e.g., user thumbs-up/down) to self-improve memory.


### More Details

Refer to the full [Update Memory API Reference](/api-reference/memory/update-memory) and [Batch Update Reference](/api-reference/memory/batch-update) for schema and advanced fields.

---

## Need help?
If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx"/>
"""
logger.info("## Tips")

logger.info("\n\n[DONE]", bright=True)