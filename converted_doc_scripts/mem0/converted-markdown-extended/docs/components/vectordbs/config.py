import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import Memory
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
title: Configurations
icon: "gear"
iconType: "solid"
---

## How to define configurations?

The `config` is defined as an object with two main keys:
- `vector_store`: Specifies the vector database provider and its configuration
  - `provider`: The name of the vector database (e.g., "chroma", "pgvector", "qdrant", "milvus", "upstash_vector", "azure_ai_search", "vertex_ai_vector_search")
  - `config`: A nested dictionary containing provider-specific settings


## How to Use Config

Here's a general example of how to use the config with mem0:

<CodeGroup>
"""
logger.info("## How to define configurations?")


# os.environ["OPENAI_API_KEY"] = "sk-xx"

config = {
    "vector_store": {
        "provider": "your_chosen_provider",
        "config": {
        }
    }
}

m = Memory.from_config(config)
m.add("Your text here", user_id="user", metadata={"category": "example"})

"""

"""


configMemory = {
  vector_store: {
    provider: 'memory',
    config: {
      collectionName: 'memories',
      dimension: 1536,
    },
  },
}

memory = new Memory(configMemory)
async def run_async_code_444a0475():
    await memory.add("Your text here", { userId: "user", metadata: { category: "example" } })
    return 
 = asyncio.run(run_async_code_444a0475())
logger.success(format_json())

"""
</CodeGroup>

<Note>
  The in-memory vector database is only supported in the TypeScript implementation.
</Note>

## Why is Config Needed?

Config is essential for:
1. Specifying which vector database to use.
2. Providing necessary connection details (e.g., host, port, credentials).
3. Customizing database-specific settings (e.g., collection name, path).
4. Ensuring proper initialization and connection to your chosen vector store.

## Master List of All Params in Config

Here's a comprehensive list of all parameters that can be used across different vector databases:

<Tabs>
<Tab title="Python">
| Parameter | Description |
|-----------|-------------|
| `collection_name` | Name of the collection |
| `embedding_model_dims` | Dimensions of the embedding model |
| `client` | Custom client for the database |
| `path` | Path for the database |
| `host` | Host where the server is running |
| `port` | Port where the server is running |
| `user` | Username for database connection |
| `password` | Password for database connection |
| `dbname` | Name of the database |
| `url` | Full URL for the server |
| `api_key` | API key for the server |
| `on_disk` | Enable persistent storage |
| `endpoint_id` | Endpoint ID (vertex_ai_vector_search) |
| `index_id` | Index ID (vertex_ai_vector_search) |
| `deployment_index_id` | Deployment index ID (vertex_ai_vector_search) |
| `project_id` | Project ID (vertex_ai_vector_search) |
| `project_number` | Project number (vertex_ai_vector_search) |
| `vector_search_api_endpoint` | Vector search API endpoint (vertex_ai_vector_search) |
| `connection_string` | PostgreSQL connection string (for Supabase/PGVector) |
| `index_method` | Vector index method (for Supabase) |
| `index_measure` | Distance measure for similarity search (for Supabase) |
</Tab>
<Tab title="TypeScript">
| Parameter | Description |
|-----------|-------------|
| `collectionName` | Name of the collection |
| `embeddingModelDims` | Dimensions of the embedding model |
| `dimension` | Dimensions of the embedding model (for memory provider) |
| `host` | Host where the server is running |
| `port` | Port where the server is running |
| `url` | URL for the server |
| `apiKey` | API key for the server |
| `path` | Path for the database |
| `onDisk` | Enable persistent storage |
| `redisUrl` | URL for the Redis server |
| `username` | Username for database connection |
| `password` | Password for database connection |
</Tab>
</Tabs>

## Customizing Config

Each vector database has its own specific configuration requirements. To customize the config for your chosen vector store:

1. Identify the vector database you want to use from [supported vector databases](./dbs).
2. Refer to the `Config` section in the respective vector database's documentation.
3. Include only the relevant parameters for your chosen database in the `config` dictionary.

## Supported Vector Databases

For detailed information on configuring specific vector databases, please visit the [Supported Vector Databases](./dbs) section. There you'll find individual pages for each supported vector store with provider-specific usage examples and configuration details.
"""
logger.info("## Why is Config Needed?")

logger.info("\n\n[DONE]", bright=True)