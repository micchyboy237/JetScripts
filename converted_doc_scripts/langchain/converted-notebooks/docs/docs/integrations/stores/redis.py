from jet.logger import logger
from langchain_community.storage import RedisStore
import os
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
---
sidebar_label: Redis
---

# RedisStore

This will help you get started with Redis [key-value stores](/docs/concepts/key_value_stores). For detailed documentation of all `RedisStore` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/storage/langchain_community.storage.redis.RedisStore.html).

## Overview

The `RedisStore` is an implementation of `ByteStore` that stores everything in your Redis instance.

### Integration details

| Class | Package | Local | [JS support](https://js.langchain.com/docs/integrations/stores/ioredis_storage) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: |
| [RedisStore](https://python.langchain.com/api_reference/community/storage/langchain_community.storage.redis.RedisStore.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |

## Setup

To create a Redis byte store, you'll need to set up a Redis instance. You can do this locally or via a provider - see our [Redis guide](/docs/integrations/providers/redis) for an overview of options.

### Installation

The LangChain `RedisStore` integration lives in the `langchain-community` package:
"""
logger.info("# RedisStore")

# %pip install -qU langchain-community redis

"""
## Instantiation

Now we can instantiate our byte store:
"""
logger.info("## Instantiation")


kv_store = RedisStore(redis_url="redis://localhost:6379")

"""
## Usage

You can set data under keys like this using the `mset` method:
"""
logger.info("## Usage")

kv_store.mset(
    [
        ["key1", b"value1"],
        ["key2", b"value2"],
    ]
)

kv_store.mget(
    [
        "key1",
        "key2",
    ]
)

"""
And you can delete data using the `mdelete` method:
"""
logger.info("And you can delete data using the `mdelete` method:")

kv_store.mdelete(
    [
        "key1",
        "key2",
    ]
)

kv_store.mget(
    [
        "key1",
        "key2",
    ]
)

"""
## API reference

For detailed documentation of all `RedisStore` features and configurations, head to the API reference: https://python.langchain.com/api_reference/community/storage/langchain_community.storage.redis.RedisStore.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)