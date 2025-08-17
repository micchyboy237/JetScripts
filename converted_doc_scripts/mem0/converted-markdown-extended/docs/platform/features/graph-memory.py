import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import MemoryClient
import os
import shutil
import { MemoryClient } from "mem0"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Graph Memory
icon: "circle-nodes"
iconType: "solid"
description: "Enable graph-based memory retrieval for more contextually relevant results"
---

## Overview

Graph Memory enhances memory pipeline by creating relationships between entities in your data. It builds a network of interconnected information for more contextually relevant search results.

This feature allows your AI applications to understand connections between entities, providing richer context for responses. It's ideal for applications needing relationship tracking and nuanced information retrieval across related memories.

## How Graph Memory Works

The Graph Memory feature analyzes how each entity connects and relates to each other. When enabled:

1. Mem0 automatically builds a graph representation of entities
2. Retrieval considers graph relationships between entities
3. Results include entities that may be contextually important even if they're not direct semantic matches

## Using Graph Memory

To use Graph Memory, you need to enable it in your API calls by setting the `enable_graph=True` parameter. You'll also need to specify `output_format="v1.1"` to receive the enriched response format.

### Adding Memories with Graph Memory

When adding new memories, enable Graph Memory to automatically build relationships with existing memories:

<CodeGroup>
"""
logger.info("## Overview")


client = MemoryClient(
    api_key="your-api-key",
    org_id="your-org-id",
    project_id="your-project-id"
)

messages = [
    {"role": "user", "content": "My name is Joseph"},
    {"role": "assistant", "content": "Hello Joseph, it's nice to meet you!"},
    {"role": "user", "content": "I'm from Seattle and I work as a software engineer"}
]

client.add(
    messages,
    user_id="joseph",
    version="v1",
    enable_graph=True,
    output_format="v1.1"
)

"""

"""


client = new MemoryClient({
  apiKey: "your-api-key",
  org_id: "your-org-id",
  project_id: "your-project-id"
})

messages = [
  { role: "user", content: "My name is Joseph" },
  { role: "assistant", content: "Hello Joseph, it's nice to meet you!" },
  { role: "user", content: "I'm from Seattle and I work as a software engineer" }
]

async def run_async_code_fd191e1b():
    await client.add({
    return 
 = asyncio.run(run_async_code_fd191e1b())
logger.success(format_json())
  messages,
  user_id: "joseph",
  version: "v1",
  enable_graph: true,
  output_format: "v1.1"
})

"""

"""

{
  "results": [
    {
      "memory": "Name is Joseph",
      "event": "ADD",
      "id": "4a5a417a-fa10-43b5-8c53-a77c45e80438"
    },
    {
      "memory": "Is from Seattle",
      "event": "ADD",
      "id": "8d268d0f-5452-4714-b27d-ae46f676a49d"
    },
    {
      "memory": "Is a software engineer",
      "event": "ADD",
      "id": "5f0a184e-ddea-4fe6-9b92-692d6a901df8"
    }
  ]
}

"""
</CodeGroup>

The graph memory would look like this:

<Frame>
  <img src="/images/graph-platform.png" alt="Graph Memory Visualization showing relationships between entities" />
</Frame>

<Caption>Graph Memory creates a network of relationships between entities, enabling more contextual retrieval</Caption>


<Note>
Response for the graph memory's `add` operation will not be available directly in the response. 
As adding graph memories is an asynchronous operation due to heavy processing, 
you can use the `get_all()` endpoint to retrieve the memory with the graph metadata.
</Note>


### Searching with Graph Memory

When searching memories, Graph Memory helps retrieve entities that are contextually important even if they're not direct semantic matches.

<CodeGroup>
"""
logger.info("### Searching with Graph Memory")

results = client.search(
    "what is my name?",
    user_id="joseph",
    enable_graph=True,
    output_format="v1.1"
)

logger.debug(results)

"""

"""

async def run_async_code_b9931602():
    results = await client.search({
    return results
results = asyncio.run(run_async_code_b9931602())
logger.success(format_json(results))
  query: "what is my name?",
  user_id: "joseph",
  enable_graph: true,
  output_format: "v1.1"
})

console.log(results)

"""

"""

{
  "results": [
    {
      "id": "4a5a417a-fa10-43b5-8c53-a77c45e80438",
      "memory": "Name is Joseph",
      "user_id": "joseph",
      "metadata": null,
      "categories": ["personal_details"],
      "immutable": false,
      "created_at": "2025-03-19T09:09:00.146390-07:00",
      "updated_at": "2025-03-19T09:09:00.146404-07:00",
      "score": 0.3621795393335552
    },
    {
      "id": "8d268d0f-5452-4714-b27d-ae46f676a49d",
      "memory": "Is from Seattle",
      "user_id": "joseph",
      "metadata": null,
      "categories": ["personal_details"],
      "immutable": false,
      "created_at": "2025-03-19T09:09:00.170680-07:00",
      "updated_at": "2025-03-19T09:09:00.170692-07:00",
      "score": 0.31212713194651254
    }
  ],
  "relations": [
    {
      "source": "joseph",
      "source_type": "person",
      "relationship": "name",
      "target": "joseph",
      "target_type": "person",
      "score": 0.39
    }
  ]
}

"""
</CodeGroup>

### Retrieving All Memories with Graph Memory

When retrieving all memories, Graph Memory provides additional relationship context:

<CodeGroup>
"""
logger.info("### Retrieving All Memories with Graph Memory")

memories = client.get_all(
    user_id="joseph",
    enable_graph=True,
    output_format="v1.1"
)

logger.debug(memories)

"""

"""

async def run_async_code_4187814c():
    memories = await client.getAll({
    return memories
memories = asyncio.run(run_async_code_4187814c())
logger.success(format_json(memories))
  user_id: "joseph",
  enable_graph: true,
  output_format: "v1.1"
})

console.log(memories)

"""

"""

{
  "results": [
    {
      "id": "5f0a184e-ddea-4fe6-9b92-692d6a901df8",
      "memory": "Is a software engineer",
      "user_id": "joseph",
      "metadata": null,
      "categories": ["professional_details"],
      "immutable": false,
      "created_at": "2025-03-19T09:09:00.194116-07:00",
      "updated_at": "2025-03-19T09:09:00.194128-07:00",
    },
    {
      "id": "8d268d0f-5452-4714-b27d-ae46f676a49d",
      "memory": "Is from Seattle",
      "user_id": "joseph",
      "metadata": null,
      "categories": ["personal_details"],
      "immutable": false,
      "created_at": "2025-03-19T09:09:00.170680-07:00",
      "updated_at": "2025-03-19T09:09:00.170692-07:00",
    },
    {
      "id": "4a5a417a-fa10-43b5-8c53-a77c45e80438",
      "memory": "Name is Joseph",
      "user_id": "joseph",
      "metadata": null,
      "categories": ["personal_details"],
      "immutable": false,
      "created_at": "2025-03-19T09:09:00.146390-07:00",
      "updated_at": "2025-03-19T09:09:00.146404-07:00",
    }
  ],
  "relations": [
    {
      "source": "joseph",
      "source_type": "person",
      "relationship": "name",
      "target": "joseph",
      "target_type": "person"
    },
    {
      "source": "joseph",
      "source_type": "person",
      "relationship": "city",
      "target": "seattle",
      "target_type": "city"
    },
    {
      "source": "joseph",
      "source_type": "person",
      "relationship": "job",
      "target": "software engineer",
      "target_type": "job"
    }
  ]
}

"""
</CodeGroup>

### Setting Graph Memory at Project Level

Instead of passing `enable_graph=True` to every add call, you can enable it once at the project level:

<CodeGroup>
"""
logger.info("### Setting Graph Memory at Project Level")


client = MemoryClient(
    api_key="your-api-key",
    org_id="your-org-id",
    project_id="your-project-id"
)

client.project.update(enable_graph=True)

messages = [
    {"role": "user", "content": "My name is Joseph"},
    {"role": "assistant", "content": "Hello Joseph, it's nice to meet you!"},
    {"role": "user", "content": "I'm from Seattle and I work as a software engineer"}
]

client.add(
    messages,
    user_id="joseph",
    output_format="v1.1"
)

"""

"""


client = new MemoryClient({
  apiKey: "your-api-key",
  org_id: "your-org-id",
  project_id: "your-project-id"
})

async def run_async_code_c61f56ef():
    await client.updateProject({ enable_graph: true, version: "v1" })
    return 
 = asyncio.run(run_async_code_c61f56ef())
logger.success(format_json())

messages = [
  { role: "user", content: "My name is Joseph" },
  { role: "assistant", content: "Hello Joseph, it's nice to meet you!" },
  { role: "user", content: "I'm from Seattle and I work as a software engineer" }
]

async def run_async_code_fd191e1b():
    await client.add({
    return 
 = asyncio.run(run_async_code_fd191e1b())
logger.success(format_json())
  messages,
  user_id: "joseph",
  output_format: "v1.1"
})

"""
</CodeGroup>


## Best Practices

- Enable Graph Memory for applications where understanding context and relationships between memories is important
- Graph Memory works best with a rich history of related conversations
- Consider Graph Memory for long-running assistants that need to track evolving information

## Performance Considerations

Graph Memory requires additional processing and may increase response times slightly for very large memory stores. However, for most use cases, the improved retrieval quality outweighs the minimal performance impact.

If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Best Practices")

logger.info("\n\n[DONE]", bright=True)