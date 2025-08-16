from autogen import Cache
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# LLM Caching

AutoGen supports caching API requests so that they can be reused when the same request is issued. This is useful when repeating or continuing experiments for reproducibility and cost saving.

Since version [`0.2.8`](https://github.com/autogenhub/autogen/releases/tag/v0.2.8), a configurable context manager allows you to easily
configure LLM cache, using either [`DiskCache`](/docs/reference/cache/disk_cache#diskcache), [`RedisCache`](/docs/reference/cache/redis_cache#rediscache), or Cosmos DB Cache. All agents inside the context manager will use the same cache.
"""
logger.info("# LLM Caching")


with Cache.redis(redis_url="redis://localhost:6379/0") as cache:
    user.initiate_chat(assistant, message=coding_task, cache=cache)

with Cache.disk() as cache:
    user.initiate_chat(assistant, message=coding_task, cache=cache)

with Cache.cosmos_db(connection_string="your_connection_string", database_id="your_database_id", container_id="your_container_id") as cache:
    user.initiate_chat(assistant, message=coding_task, cache=cache)

"""
The cache can also be passed directly to the model client's create call.
"""
logger.info("The cache can also be passed directly to the model client's create call.")

client = MLXWrapper(...)
with Cache.disk() as cache:
    client.create(..., cache=cache)

"""
## Controlling the seed

You can vary the `cache_seed` parameter to get different LLM output while
still using cache.
"""
logger.info("## Controlling the seed")

with Cache.disk(cache_seed=1) as cache:
    user.initiate_chat(assistant, message=coding_task, cache=cache)

"""
## Cache path

By default [`DiskCache`](/docs/reference/cache/disk_cache#diskcache) uses `.cache` for storage. To change the cache directory,
set `cache_path_root`:
"""
logger.info("## Cache path")

with Cache.disk(cache_path_root="/tmp/autogen_cache") as cache:
    user.initiate_chat(assistant, message=coding_task, cache=cache)

"""
## Disabling cache

For backward compatibility, [`DiskCache`](/docs/reference/cache/disk_cache#diskcache) is on by default with `cache_seed` set to 41.
To disable caching completely, set `cache_seed` to `None` in the `llm_config` of the agent.
"""
logger.info("## Disabling cache")

assistant = AssistantAgent(
    "coding_agent",
    llm_config={
        "cache_seed": None,
        "config_list": OAI_CONFIG_LIST,
        "max_tokens": 1024,
    },
)

"""
## Difference between `cache_seed` and MLX's `seed` parameter

MLX v1.1 introduced a new parameter `seed`. The difference between AutoGen's `cache_seed` and MLX's `seed` is AutoGen uses an explicit request cache to guarantee the exactly same output is produced for the same input and when cache is hit, no MLX API call will be made. MLX's `seed` is a best-effort deterministic sampling with no guarantee of determinism. When using MLX's `seed` with `cache_seed` set to `None`, even for the same input, an MLX API call will be made and there is no guarantee for getting exactly the same output.
"""
logger.info("## Difference between `cache_seed` and MLX's `seed` parameter")

logger.info("\n\n[DONE]", bright=True)