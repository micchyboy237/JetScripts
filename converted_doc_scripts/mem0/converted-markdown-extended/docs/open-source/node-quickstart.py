import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
import os
import shutil
import { Memory } from 'mem0ai/oss'


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Node SDK Quickstart
description: 'Get started with Mem0 quickly!'
icon: "node"
iconType: "solid"
---

> Welcome to the Mem0 quickstart guide. This guide will help you get up and running with Mem0 in no time.

## Installation

To install Mem0, you can use npm. Run the following command in your terminal:
"""
logger.info("## Installation")

npm install mem0ai

"""
## Basic Usage

### Initialize Mem0

<Tabs>
  <Tab title="Basic">
"""
logger.info("## Basic Usage")


memory = new Memory()

"""
</Tab>
  <Tab title="Advanced">
If you want to run Mem0 in production, initialize using the following method:
"""
logger.info("If you want to run Mem0 in production, initialize using the following method:")


memory = new Memory({
    version: 'v1.1',
    embedder: {
      provider: 'openai',
      config: {
#         apiKey: process.env.OPENAI_API_KEY || '',
        model: 'mxbai-embed-large',
      },
    },
    vectorStore: {
      provider: 'memory',
      config: {
        collectionName: 'memories',
        dimension: 1536,
      },
    },
    llm: {
      provider: 'openai',
      config: {
#         apiKey: process.env.OPENAI_API_KEY || '',
        model: 'gpt-4-turbo-preview',
      },
    },
    historyDbPath: 'memory.db',
  })

"""
</Tab>
</Tabs>


### Store a Memory

<CodeGroup>
"""
logger.info("### Store a Memory")

messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]

async def run_async_code_0c676ac8():
    await memory.add(messages, { userId: "alice", metadata: { category: "movie_recommendations" } })
    return 
 = asyncio.run(run_async_code_0c676ac8())
logger.success(format_json())

"""

"""

{
  "results": [
    {
      "id": "892db2ae-06d9-49e5-8b3e-585ef9b85b8e",
      "memory": "User is planning to watch a movie tonight.",
      "metadata": {
        "category": "movie_recommendations"
      }
    },
    {
      "id": "cbb1fe73-0bf1-4067-8c1f-63aa53e7b1a4",
      "memory": "User is not a big fan of thriller movies.",
      "metadata": {
        "category": "movie_recommendations"
      }
    },
    {
      "id": "475bde34-21e6-42ab-8bef-0ab84474f156",
      "memory": "User loves sci-fi movies.",
      "metadata": {
        "category": "movie_recommendations"
      }
    }
  ]
}

"""
</CodeGroup>

### Retrieve Memories

<CodeGroup>
"""
logger.info("### Retrieve Memories")

async def run_async_code_a6b74369():
    async def run_async_code_4ccf8466():
        allMemories = await memory.getAll({ userId: "alice" })
        return allMemories
    allMemories = asyncio.run(run_async_code_4ccf8466())
    logger.success(format_json(allMemories))
    return allMemories
allMemories = asyncio.run(run_async_code_a6b74369())
logger.success(format_json(allMemories))
console.log(allMemories)

"""

"""

{
  "results": [
    {
      "id": "892db2ae-06d9-49e5-8b3e-585ef9b85b8e",
      "memory": "User is planning to watch a movie tonight.",
      "hash": "1a271c007316c94377175ee80e746a19",
      "createdAt": "2025-02-27T16:33:20.557Z",
      "updatedAt": "2025-02-27T16:33:27.051Z",
      "metadata": {
        "category": "movie_recommendations"
      },
      "userId": "alice"
    },
    {
      "id": "475bde34-21e6-42ab-8bef-0ab84474f156",
      "memory": "User loves sci-fi movies.",
      "hash": "285d07801ae42054732314853e9eadd7",
      "createdAt": "2025-02-27T16:33:20.560Z",
      "updatedAt": undefined,
      "metadata": {
        "category": "movie_recommendations"
      },
      "userId": "alice"
    },
    {
      "id": "cbb1fe73-0bf1-4067-8c1f-63aa53e7b1a4",
      "memory": "User is not a big fan of thriller movies.",
      "hash": "285d07801ae42054732314853e9eadd7",
      "createdAt": "2025-02-27T16:33:20.560Z",
      "updatedAt": undefined,
      "metadata": {
        "category": "movie_recommendations"
      },
      "userId": "alice"
    }
  ]
}

"""
</CodeGroup>


<br />

<CodeGroup>
"""

async def run_async_code_fabfa921():
    async def run_async_code_ad86dfbb():
        singleMemory = await memory.get('892db2ae-06d9-49e5-8b3e-585ef9b85b8e')
        return singleMemory
    singleMemory = asyncio.run(run_async_code_ad86dfbb())
    logger.success(format_json(singleMemory))
    return singleMemory
singleMemory = asyncio.run(run_async_code_fabfa921())
logger.success(format_json(singleMemory))
console.log(singleMemory)

"""

"""

{
  "id": "892db2ae-06d9-49e5-8b3e-585ef9b85b8e",
  "memory": "User is planning to watch a movie tonight.",
  "hash": "1a271c007316c94377175ee80e746a19",
  "createdAt": "2025-02-27T16:33:20.557Z",
  "updatedAt": undefined,
  "metadata": {
    "category": "movie_recommendations"
  },
  "userId": "alice"
}

"""
</CodeGroup>

### Search Memories

<CodeGroup>
"""
logger.info("### Search Memories")

async def run_async_code_d01bd73f():
    async def run_async_code_47a81d45():
        result = await memory.search('What do you know about me?', { userId: "alice" })
        return result
    result = asyncio.run(run_async_code_47a81d45())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_d01bd73f())
logger.success(format_json(result))
console.log(result)

"""

"""

{
  "results": [
    {
      "id": "892db2ae-06d9-49e5-8b3e-585ef9b85b8e",
      "memory": "User is planning to watch a movie tonight.",
      "hash": "1a271c007316c94377175ee80e746a19",
      "createdAt": "2025-02-27T16:33:20.557Z",
      "updatedAt": undefined,
      "score": 0.38920719231944799,
      "metadata": {
        "category": "movie_recommendations"
      },
      "userId": "alice"
    },
    {
      "id": "475bde34-21e6-42ab-8bef-0ab84474f156",
      "memory": "User loves sci-fi movies.",
      "hash": "285d07801ae42054732314853e9eadd7",
      "createdAt": "2025-02-27T16:33:20.560Z",
      "updatedAt": undefined,
      "score": 0.36869761478135689,
      "metadata": {
        "category": "movie_recommendations"
      },
      "userId": "alice"
    },
    {
      "id": "cbb1fe73-0bf1-4067-8c1f-63aa53e7b1a4",
      "memory": "User is not a big fan of thriller movies.",
      "hash": "285d07801ae42054732314853e9eadd7",
      "createdAt": "2025-02-27T16:33:20.560Z",
      "updatedAt": undefined,
      "score": 0.33855272141248272,
      "metadata": {
        "category": "movie_recommendations"
      },
      "userId": "alice"
    }
  ]
}

"""
</CodeGroup>

### Update a Memory

<CodeGroup>
"""
logger.info("### Update a Memory")

async def async_func_0():
    result = await memory.update(
      '892db2ae-06d9-49e5-8b3e-585ef9b85b8e',
      'I love India, it is my favorite country.'
    )
    return result
result = asyncio.run(async_func_0())
logger.success(format_json(result))
console.log(result)

"""

"""

{
  "message": "Memory updated successfully!"
}

"""
</CodeGroup>

### Memory History

<CodeGroup>
"""
logger.info("### Memory History")

async def run_async_code_98189619():
    async def run_async_code_874815a3():
        history = await memory.history('892db2ae-06d9-49e5-8b3e-585ef9b85b8e')
        return history
    history = asyncio.run(run_async_code_874815a3())
    logger.success(format_json(history))
    return history
history = asyncio.run(run_async_code_98189619())
logger.success(format_json(history))
console.log(history)

"""

"""

[
  {
    "id": 39,
    "memoryId": "892db2ae-06d9-49e5-8b3e-585ef9b85b8e",
    "previousValue": "User is planning to watch a movie tonight.",
    "newValue": "I love India, it is my favorite country.",
    "action": "UPDATE",
    "createdAt": "2025-02-27T16:33:20.557Z",
    "updatedAt": "2025-02-27T16:33:27.051Z",
    "isDeleted": 0
  },
  {
    "id": 37,
    "memoryId": "892db2ae-06d9-49e5-8b3e-585ef9b85b8e",
    "previousValue": null,
    "newValue": "User is planning to watch a movie tonight.",
    "action": "ADD",
    "createdAt": "2025-02-27T16:33:20.557Z",
    "updatedAt": null,
    "isDeleted": 0
  }
]

"""
</CodeGroup>

### Delete Memory
"""
logger.info("### Delete Memory")

async def run_async_code_b4198bf5():
    await memory.delete('892db2ae-06d9-49e5-8b3e-585ef9b85b8e')
    return 
 = asyncio.run(run_async_code_b4198bf5())
logger.success(format_json())

async def run_async_code_28a5d23e():
    await memory.deleteAll({ userId: "alice" })
    return 
 = asyncio.run(run_async_code_28a5d23e())
logger.success(format_json())

"""
### Reset Memory
"""
logger.info("### Reset Memory")

async def run_async_code_f8c5c5b2():
    await memory.reset(); // Reset all memories
    return 
 = asyncio.run(run_async_code_f8c5c5b2())
logger.success(format_json())

"""
### History Store

Mem0 TypeScript SDK support history stores to run on a serverless environment:

We recommend using `Supabase` as a history store for serverless environments or disable history store to run on a serverless environment.

<CodeGroup>
"""
logger.info("### History Store")


memory = new Memory({
  historyStore: {
    provider: 'supabase',
    config: {
      supabaseUrl: process.env.SUPABASE_URL || '',
      supabaseKey: process.env.SUPABASE_KEY || '',
      tableName: 'memory_history',
    },
  },
})

"""

"""


memory = new Memory({
  disableHistory: true,
})

"""
</CodeGroup>

Mem0 uses SQLite as a default history store.

#### Create Memory History Table in Supabase

You may need to create a memory history table in Supabase to store the history of memories. Use the following SQL command in `SQL Editor` on the Supabase project dashboard to create a memory history table:
"""
logger.info("#### Create Memory History Table in Supabase")

create table memory_history (
  id text primary key,
  memory_id text not null,
  previous_value text,
  new_value text,
  action text not null,
  created_at timestamp with time zone default timezone('utc', now()),
  updated_at timestamp with time zone,
  is_deleted integer default 0
)

"""
## Configuration Parameters

Mem0 offers extensive configuration options to customize its behavior according to your needs. These configurations span across different components like vector stores, language models, embedders, and graph stores.

<AccordionGroup>
<Accordion title="Vector Store Configuration">
| Parameter    | Description                     | Default     |
|-------------|---------------------------------|-------------|
| `provider`   | Vector store provider (e.g., "memory") | "memory"   |
| `host`       | Host address                    | "localhost" |
| `port`       | Port number                     | undefined       |
</Accordion>

<Accordion title="LLM Configuration">
| Parameter              | Description                                   | Provider          |
|-----------------------|-----------------------------------------------|-------------------|
| `provider`            | LLM provider (e.g., "openai", "anthropic")    | All              |
| `model`               | Model to use                                  | All              |
| `temperature`         | Temperature of the model                      | All              |
| `apiKey`             | API key to use                                | All              |
| `maxTokens`          | Tokens to generate                            | All              |
| `topP`               | Probability threshold for nucleus sampling    | All              |
| `topK`               | Number of highest probability tokens to keep  | All              |
| `openaiBaseUrl`     | Base URL for MLX API                      | MLX           |
</Accordion>

<Accordion title="Graph Store Configuration">
| Parameter    | Description                     | Default     |
|-------------|---------------------------------|-------------|
| `provider`   | Graph store provider (e.g., "neo4j") | "neo4j"    |
| `url`        | Connection URL                  | env.NEO4J_URL        |
| `username`   | Authentication username         | env.NEO4J_USERNAME        |
| `password`   | Authentication password         | env.NEO4J_PASSWORD        |
</Accordion>

<Accordion title="Embedder Configuration">
| Parameter    | Description                     | Default                      |
|-------------|---------------------------------|------------------------------|
| `provider`   | Embedding provider              | "openai"                     |
| `model`      | Embedding model to use          | "mxbai-embed-large"     |
| `apiKey`    | API key for embedding service   | None                        |
</Accordion>

<Accordion title="General Configuration">
| Parameter         | Description                          | Default                    |
|------------------|--------------------------------------|----------------------------|
| `historyDbPath` | Path to the history database         | "{mem0_dir}/history.db"    |
| `version`         | API version                          | "v1.0"                     |
| `customPrompt`   | Custom prompt for memory processing  | None                       |
</Accordion>

<Accordion title="History Table Configuration">
| Parameter         | Description                          | Default                    |
|------------------|--------------------------------------|----------------------------|
| `provider`       | History store provider               | "sqlite"                   |
| `config`         | History store configuration         | None (Defaults to SQLite)                      |
| `disableHistory` | Disable history store               | false                      |
</Accordion>

<Accordion title="Complete Configuration Example">
"""
logger.info("## Configuration Parameters")

config = {
      version: 'v1.1',
      embedder: {
        provider: 'openai',
        config: {
#           apiKey: process.env.OPENAI_API_KEY || '',
          model: 'mxbai-embed-large',
        },
      },
      vectorStore: {
        provider: 'memory',
        config: {
          collectionName: 'memories',
          dimension: 1536,
        },
      },
      llm: {
        provider: 'openai',
        config: {
#           apiKey: process.env.OPENAI_API_KEY || '',
          model: 'gpt-4-turbo-preview',
        },
      },
      historyStore: {
        provider: 'supabase',
        config: {
          supabaseUrl: process.env.SUPABASE_URL || '',
          supabaseKey: process.env.SUPABASE_KEY || '',
          tableName: 'memories',
        },
      },
      disableHistory: false, // This is false by default
      customPrompt: "I'm a virtual assistant. I'm here to help you with your queries.",
    }

"""
</Accordion>
</AccordionGroup>

If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("If you have any questions, please feel free to reach out to us using one of the following methods:")

logger.info("\n\n[DONE]", bright=True)