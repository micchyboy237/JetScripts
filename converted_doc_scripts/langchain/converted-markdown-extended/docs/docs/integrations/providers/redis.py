from jet.logger import logger
from langchain.cache import RedisCache
from langchain.cache import RedisSemanticCache
from langchain.globals import set_llm_cache
from langchain_community.vectorstores import Redis
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
import os
import redis
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
# Redis

>[Redis (Remote Dictionary Server)](https://en.wikipedia.org/wiki/Redis) is an open-source in-memory storage,
> used as a distributed, in-memory key–value database, cache and message broker, with optional durability.
> Because it holds all data in memory and because of its design, `Redis` offers low-latency reads and writes,
> making it particularly suitable for use cases that require a cache. Redis is the most popular NoSQL database,
> and one of the most popular databases overall.

This page covers how to use the [Redis](https://redis.com) ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Redis wrappers.

## Installation and Setup

Install the Python SDK:
"""
logger.info("# Redis")

pip install redis

"""
To run Redis locally, you can use Docker:
"""
logger.info("To run Redis locally, you can use Docker:")

docker run --name langchain-redis -d -p 6379:6379 redis redis-server --save 60 1 --loglevel warning

"""
To stop the container:
"""
logger.info("To stop the container:")

docker stop langchain-redis

"""
And to start it again:
"""
logger.info("And to start it again:")

docker start langchain-redis

"""
### Connections

We need a redis url connection string to connect to the database support either a stand alone Redis server
or a High-Availability setup with Replication and Redis Sentinels.

#### Redis Standalone connection url
For standalone `Redis` server, the official redis connection url formats can be used as describe in the python redis modules
"from_url()" method [Redis.from_url](https://redis-py.readthedocs.io/en/stable/connections.html#redis.Redis.from_url)

Example: `redis_url = "redis://:secret-pass@localhost:6379/0"`

#### Redis Sentinel connection url

For [Redis sentinel setups](https://redis.io/docs/management/sentinel/) the connection scheme is "redis+sentinel".
This is an unofficial extensions to the official IANA registered protocol schemes as long as there is no connection url
for Sentinels available.

Example: `redis_url = "redis+sentinel://:secret-pass@sentinel-host:26379/mymaster/0"`

The format is  `redis+sentinel://[[username]:[password]]@[host-or-ip]:[port]/[service-name]/[db-number]`
with the default values of "service-name = mymaster" and "db-number = 0" if not set explicit.
The service-name is the redis server monitoring group name as configured within the Sentinel.

The current url format limits the connection string to one sentinel host only (no list can be given) and
both Redis server and sentinel must have the same password set (if used).

#### Redis Cluster connection url

Redis cluster is not supported right now for all methods requiring a "redis_url" parameter.
The only way to use a Redis Cluster is with LangChain classes accepting a preconfigured Redis client like `RedisCache`
(example below).

## Cache

The Cache wrapper allows for [Redis](https://redis.io) to be used as a remote, low-latency, in-memory cache for LLM prompts and responses.

### Standard Cache
The standard cache is the Redis bread & butter of use case in production for both [open-source](https://redis.io) and [enterprise](https://redis.com) users globally.
"""
logger.info("### Connections")


"""
To use this cache with your LLMs:
"""
logger.info("To use this cache with your LLMs:")


redis_client = redis.Redis.from_url(...)
set_llm_cache(RedisCache(redis_client))

"""
### Semantic Cache
Semantic caching allows users to retrieve cached prompts based on semantic similarity between the user input and previously cached results. Under the hood it blends Redis as both a cache and a vectorstore.
"""
logger.info("### Semantic Cache")


"""
To use this cache with your LLMs:
"""
logger.info("To use this cache with your LLMs:")



redis_url = "redis://localhost:6379"

set_llm_cache(RedisSemanticCache(
    embedding=FakeEmbeddings(),
    redis_url=redis_url
))

"""
## VectorStore

The vectorstore wrapper turns Redis into a low-latency [vector database](https://redis.com/solutions/use-cases/vector-database/) for semantic search or LLM content retrieval.
"""
logger.info("## VectorStore")


"""
For a more detailed walkthrough of the Redis vectorstore wrapper, see [this notebook](/docs/integrations/vectorstores/redis).

## Retriever

The Redis vector store retriever wrapper generalizes the vectorstore class to perform
low-latency document retrieval. To create the retriever, simply
call `.as_retriever()` on the base vectorstore class.

## Memory

Redis can be used to persist LLM conversations.

### Vector Store Retriever Memory

For a more detailed walkthrough of the `VectorStoreRetrieverMemory` wrapper, see [this notebook](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.vectorstore.VectorStoreRetrieverMemory.html).

### Chat Message History Memory
For a detailed example of Redis to cache conversation message history, see [this notebook](/docs/integrations/memory/redis_chat_message_history).
"""
logger.info("## Retriever")

logger.info("\n\n[DONE]", bright=True)