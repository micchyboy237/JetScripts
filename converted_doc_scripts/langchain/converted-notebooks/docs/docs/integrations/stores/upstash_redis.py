from jet.logger import logger
from langchain_community.storage import UpstashRedisByteStore
from upstash_redis import Redis
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
sidebar_label: Upstash Redis
---

# UpstashRedisByteStore

This will help you get started with Upstash redis [key-value stores](/docs/concepts/key_value_stores). For detailed documentation of all `UpstashRedisByteStore` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/storage/langchain_community.storage.upstash_redis.UpstashRedisByteStore.html).

## Overview

The `UpstashRedisStore` is an implementation of `ByteStore` that stores everything in your [Upstash](https://upstash.com/)-hosted Redis instance.

To use the base `RedisStore` instead, see [this guide](/docs/integrations/stores/redis/).

### Integration details

| Class | Package | Local | [JS support](https://js.langchain.com/docs/integrations/stores/upstash_redis_storage) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: |
| [UpstashRedisByteStore](https://python.langchain.com/api_reference/community/storage/langchain_community.storage.upstash_redis.UpstashRedisByteStore.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ❌ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |

## Setup

You'll first need to [sign up for an Upstash account](https://upstash.com/docs/redis/overall/getstarted). Next, you'll need to create a Redis database to connect to.

### Credentials

Once you've created your database, get your database URL (don't forget the `https://`!) and token:
"""
logger.info("# UpstashRedisByteStore")

# from getpass import getpass

# URL = getpass("Enter your Upstash URL")
# TOKEN = getpass("Enter your Upstash REST token")

"""
### Installation

The LangChain Upstash integration lives in the `langchain-community` package. You'll also need to install the `upstash-redis` package as a peer dependency:
"""
logger.info("### Installation")

# %pip install -qU langchain-community upstash-redis

"""
## Instantiation

Now we can instantiate our byte store:
"""
logger.info("## Instantiation")


redis_client = Redis(url=URL, token=TOKEN)
kv_store = UpstashRedisByteStore(client=redis_client, ttl=None, namespace="test-ns")

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

For detailed documentation of all `UpstashRedisByteStore` features and configurations, head to the API reference: https://python.langchain.com/api_reference/community/storage/langchain_community.storage.upstash_redis.UpstashRedisByteStore.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)