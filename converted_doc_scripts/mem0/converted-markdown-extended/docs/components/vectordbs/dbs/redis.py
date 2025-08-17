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
[Redis](https://redis.io/) is a scalable, real-time database that can store, search, and analyze vector data.

### Installation
"""
logger.info("### Installation")

pip install redis redisvl

"""
Redis Stack using Docker:
"""
logger.info("Redis Stack using Docker:")

docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

"""
### Usage

<CodeGroup>
"""
logger.info("### Usage")


# os.environ["OPENAI_API_KEY"] = "sk-xx"

config = {
    "vector_store": {
        "provider": "redis",
        "config": {
            "collection_name": "mem0",
            "embedding_model_dims": 1536,
            "redis_url": "redis://localhost:6379"
        }
    },
    "version": "v1.1"
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""

"""


config = {
  vectorStore: {
    provider: 'redis',
    config: {
      collectionName: 'memories',
      embeddingModelDims: 1536,
      redisUrl: 'redis://localhost:6379',
      username: 'your-redis-username',
      password: 'your-redis-password',
    },
  },
}

memory = new Memory(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
async def run_async_code_3ea57109():
    await memory.add(messages, { userId: "alice", metadata: { category: "movies" } })
    return 
 = asyncio.run(run_async_code_3ea57109())
logger.success(format_json())

"""
</CodeGroup>

### Config

Let's see the available parameters for the `redis` config:

<Tabs>
<Tab title="Python">
| Parameter | Description | Default Value |
| --- | --- | --- |
| `collection_name` | The name of the collection to store the vectors | `mem0` |
| `embedding_model_dims` | Dimensions of the embedding model | `1536` |
| `redis_url` | The URL of the Redis server | `None` |
</Tab>
<Tab title="TypeScript">
| Parameter | Description | Default Value |
| --- | --- | --- |
| `collectionName` | The name of the collection to store the vectors | `mem0` |
| `embeddingModelDims` | Dimensions of the embedding model | `1536` |
| `redisUrl` | The URL of the Redis server | `None` |
| `username` | Username for Redis connection | `None` |
| `password` | Password for Redis connection | `None` |
</Tab>
</Tabs>
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)