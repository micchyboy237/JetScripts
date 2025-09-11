from jet.adapters.langchain.chat_ollama import Ollama, OllamaEmbeddings
from jet.logger import logger
from langchain.globals import set_llm_cache
from langchain.schema import Generation
from langchain_redis import RedisCache, RedisSemanticCache
import jet.adapters.langchain.chat_ollama
import langchain_core
import ollama
import os
import redis
import shutil
import time


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
# Redis Cache for LangChain

This notebook demonstrates how to use the `RedisCache` and `RedisSemanticCache` classes from the langchain-redis package to implement caching for LLM responses.

## Setup

First, let's install the required dependencies and ensure we have a Redis instance running.
"""
logger.info("# Redis Cache for LangChain")

# %pip install -U langchain-core langchain-redis langchain-ollama redis

"""
Ensure you have a Redis server running. You can start one using Docker with:

```
docker run -d -p 6379:6379 redis:latest
```

Or install and run Redis locally according to your operating system's instructions.
"""
logger.info("Ensure you have a Redis server running. You can start one using Docker with:")


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
logger.debug(f"Connecting to Redis at: {REDIS_URL}")

"""
## Importing Required Libraries
"""
logger.info("## Importing Required Libraries")




"""
### Set Ollama API key
"""
logger.info("### Set Ollama API key")

# from getpass import getpass

# ollama_api_key = os.getenv("OPENAI_API_KEY")

if not ollama_api_key:
    logger.debug("Ollama API key not found in environment variables.")
#     ollama_api_key = getpass("Please enter your Ollama API key: ")

#     os.environ["OPENAI_API_KEY"] = ollama_api_key
    logger.debug("Ollama API key has been set for this session.")
else:
    logger.debug("Ollama API key found in environment variables.")

"""
## Using RedisCache
"""
logger.info("## Using RedisCache")

redis_cache = RedisCache(redis_url=REDIS_URL)

set_llm_cache(redis_cache)

llm = Ollama(temperature=0)


def timed_completion(prompt):
    start_time = time.time()
    result = llm.invoke(prompt)
    end_time = time.time()
    return result, end_time - start_time


prompt = "Explain the concept of caching in three sentences."
result1, time1 = timed_completion(prompt)
logger.debug(f"First call (not cached):\nResult: {result1}\nTime: {time1:.2f} seconds\n")

result2, time2 = timed_completion(prompt)
logger.debug(f"Second call (cached):\nResult: {result2}\nTime: {time2:.2f} seconds\n")

logger.debug(f"Speed improvement: {time1 / time2:.2f}x faster")

redis_cache.clear()
logger.debug("Cache cleared")

"""
## Using RedisSemanticCache
"""
logger.info("## Using RedisSemanticCache")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
semantic_cache = RedisSemanticCache(
    redis_url=REDIS_URL, embeddings=embeddings, distance_threshold=0.2
)

set_llm_cache(semantic_cache)


def test_semantic_cache(prompt):
    start_time = time.time()
    result = llm.invoke(prompt)
    end_time = time.time()
    return result, end_time - start_time


original_prompt = "What is the capital of France?"
result1, time1 = test_semantic_cache(original_prompt)
logger.debug(
    f"Original query:\nPrompt: {original_prompt}\nResult: {result1}\nTime: {time1:.2f} seconds\n"
)

similar_prompt = "Can you tell me the capital city of France?"
result2, time2 = test_semantic_cache(similar_prompt)
logger.debug(
    f"Similar query:\nPrompt: {similar_prompt}\nResult: {result2}\nTime: {time2:.2f} seconds\n"
)

logger.debug(f"Speed improvement: {time1 / time2:.2f}x faster")

semantic_cache.clear()
logger.debug("Semantic cache cleared")

"""
## Advanced Usage

### Custom TTL (Time-To-Live)
"""
logger.info("## Advanced Usage")

ttl_cache = RedisCache(redis_url=REDIS_URL, ttl=5)  # 60 seconds TTL

ttl_cache.update("test_prompt", "test_llm", [Generation(text="Cached response")])

cached_result = ttl_cache.lookup("test_prompt", "test_llm")
logger.debug(f"Cached result: {cached_result[0].text if cached_result else 'Not found'}")

logger.debug("Waiting for TTL to expire...")
time.sleep(6)

expired_result = ttl_cache.lookup("test_prompt", "test_llm")
logger.debug(
    f"Result after TTL: {expired_result[0].text if expired_result else 'Not found (expired)'}"
)

"""
### Customizing RedisSemanticCache
"""
logger.info("### Customizing RedisSemanticCache")

custom_semantic_cache = RedisSemanticCache(
    redis_url=REDIS_URL,
    embeddings=embeddings,
    distance_threshold=0.1,  # Stricter similarity threshold
    ttl=3600,  # 1 hour TTL
    name="custom_cache",  # Custom cache name
)

set_llm_cache(custom_semantic_cache)

test_prompt = "What's the largest planet in our solar system?"
result, _ = test_semantic_cache(test_prompt)
logger.debug(f"Original result: {result}")

similar_test_prompt = "Which planet is the biggest in the solar system?"
similar_result, _ = test_semantic_cache(similar_test_prompt)
logger.debug(f"Similar query result: {similar_result}")

custom_semantic_cache.clear()

"""
## Conclusion

This notebook demonstrated the usage of `RedisCache` and `RedisSemanticCache` from the langchain-redis package. These caching mechanisms can significantly improve the performance of LLM-based applications by reducing redundant API calls and leveraging semantic similarity for intelligent caching. The Redis-based implementation provides a fast, scalable, and flexible solution for caching in distributed systems.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)