from jet.logger import CustomLogger
from mem0 import MemoryClient
import MemoryClient from 'mem0ai'
import MemoryClient, { Message, MemoryOptions } from 'mem0ai'
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
title: Quickstart
description: 'Get started with Mem0 Platform in minutes'
icon: "bolt"
iconType: "solid"
---

Get up and running with Mem0 Platform quickly. This guide covers the essential steps to start storing and retrieving memories.

## 1. Installation

<CodeGroup>
"""
logger.info("## 1. Installation")

pip install mem0ai

"""

"""

npm install mem0ai

"""
</CodeGroup>

## 2. API Key Setup

1. Sign in to [Mem0 Platform](https://mem0.dev/pd-api)
2. Copy your API Key from the dashboard

![Get API Key from Mem0 Platform](/images/platform/api-key.png)

## 3. Initialize Client

<CodeGroup>
"""
logger.info("## 2. API Key Setup")


os.environ["MEM0_API_KEY"] = "your-api-key"
client = MemoryClient()

"""

"""

client = new MemoryClient({ apiKey: 'your-api-key' })

"""
</CodeGroup>

## 4. Basic Operations

### Add Memories

Store user preferences and context:

<CodeGroup>
"""
logger.info("## 4. Basic Operations")

messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and allergic to nuts."},
    {"role": "assistant", "content": "Hello Alex! I'll remember your dietary preferences."}
]

result = client.add(messages, user_id="alex")
logger.debug(result)

"""

"""

messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and allergic to nuts."},
    {"role": "assistant", "content": "Hello Alex! I'll remember your dietary preferences."}
]

client.add(messages, { user_id: "alex" })
    .then(result => console.log(result))
    .catch(error => console.error(error))

"""
</CodeGroup>

### Search Memories

Retrieve relevant memories based on queries:

<CodeGroup>
"""
logger.info("### Search Memories")

query = "What should I cook for dinner?"
results = client.search(query, user_id="alex")
logger.debug(results)

"""

"""

query = "What should I cook for dinner?"
client.search(query, { user_id: "alex" })
    .then(results => console.log(results))
    .catch(error => console.error(error))

"""
</CodeGroup>

### Get All Memories

Fetch all memories for a user:

<CodeGroup>
"""
logger.info("### Get All Memories")

memories = client.get_all(user_id="alex")
logger.debug(memories)

"""

"""

client.getAll({ user_id: "alex" })
    .then(memories => console.log(memories))
    .catch(error => console.error(error))

"""
</CodeGroup>

## 5. Memory Types

### User Memories
Long-term memories that persist across sessions:

<CodeGroup>
"""
logger.info("## 5. Memory Types")

client.add(messages, user_id="alex", metadata={"category": "preferences"})

"""

"""

client.add(messages, { user_id: "alex", metadata: { category: "preferences" } })

"""
</CodeGroup>

### Session Memories
Short-term memories for specific conversations:

<CodeGroup>
"""
logger.info("### Session Memories")

client.add(messages, user_id="alex", run_id="session-123")

"""

"""

client.add(messages, { user_id: "alex", run_id: "session-123" })

"""
</CodeGroup>

### Agent Memories
Memories for AI assistants and agents:

<CodeGroup>
"""
logger.info("### Agent Memories")

client.add(messages, agent_id="support-bot")

"""

"""

client.add(messages, { agent_id: "support-bot" })

"""
</CodeGroup>

## 6. Advanced Features

### Async Processing
Process memories in the background for faster responses:

<CodeGroup>
"""
logger.info("## 6. Advanced Features")

client.add(messages, user_id="alex", async_mode=True)

"""

"""

client.add(messages, { user_id: "alex", async_mode: true })

"""
</CodeGroup>

### Search with Filters
Filter results by categories and metadata:

<CodeGroup>
"""
logger.info("### Search with Filters")

results = client.search(
    "food preferences",
    user_id="alex",
    categories=["preferences"],
    metadata={"category": "food"}
)

"""

"""

client.search("food preferences", {
    user_id: "alex",
    categories: ["preferences"],
    metadata: { category: "food" }
})

"""
</CodeGroup>

## TypeScript Example

<CodeGroup>
"""
logger.info("## TypeScript Example")


client = new MemoryClient('your-api-key')

messages: Message[] = [
    { role: "user", content: "I love Italian food" },
    { role: "assistant", content: "Noted! I'll remember your preference for Italian cuisine." }
]

options: MemoryOptions = {
    user_id: "alex",
    metadata: { category: "food_preferences" }
}

client.add(messages, options)
    .then(result => console.log(result))
    .catch(error => console.error(error))

"""
</CodeGroup>

## Next Steps

Now that you're up and running, explore more advanced features:

- **[Advanced Memory Operations](/core-concepts/memory-operations)** - Learn about filtering, updating, and managing memories
- **[Platform Features](/platform/features/platform-overview)** - Discover advanced platform capabilities
- **[API Reference](/api-reference)** - Complete API documentation

<Snippet file="get-help.mdx" />
"""
logger.info("## Next Steps")

logger.info("\n\n[DONE]", bright=True)