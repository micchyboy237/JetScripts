from jet.logger import logger
from langchain_astradb import AstraDBByteStore
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
sidebar_label: AstraDB
---

# AstraDBByteStore

This will help you get started with Astra DB [key-value stores](/docs/concepts/key_value_stores). For detailed documentation of all `AstraDBByteStore` features and configurations head to the [API reference](https://python.langchain.com/api_reference/astradb/storage/langchain_astradb.storage.AstraDBByteStore.html).

## Overview

> [DataStax Astra DB](https://docs.datastax.com/en/astra-db-serverless/index.html) is a serverless 
> AI-ready database built on `Apache Cassandra®` and made conveniently available 
> through an easy-to-use JSON API.

### Integration details

| Class | Package | Local | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: |
| [AstraDBByteStore](https://python.langchain.com/api_reference/astradb/storage/langchain_astradb.storage.AstraDBByteStore.html) | [langchain-astradb](https://python.langchain.com/api_reference/astradb/index.html) | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_astradb?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_astradb?style=flat-square&label=%20) |

## Setup

To create an `AstraDBByteStore` byte store, you'll need to [create a DataStax account](https://www.datastax.com/products/datastax-astra).

### Credentials

After signing up, set the following credentials:
"""
logger.info("# AstraDBByteStore")

# from getpass import getpass

# ASTRA_DB_API_ENDPOINT = getpass("ASTRA_DB_API_ENDPOINT = ")
# ASTRA_DB_APPLICATION_TOKEN = getpass("ASTRA_DB_APPLICATION_TOKEN = ")

"""
### Installation

The LangChain AstraDB integration lives in the `langchain-astradb` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-astradb

"""
## Instantiation

Now we can instantiate our byte store:
"""
logger.info("## Instantiation")


kv_store = AstraDBByteStore(
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    collection_name="my_store",
)

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
You can use an `AstraDBByteStore` anywhere you'd use other ByteStores, including as a [cache for embeddings](/docs/how_to/caching_embeddings).

## API reference

For detailed documentation of all `AstraDBByteStore` features and configurations, head to the API reference: https://python.langchain.com/api_reference/astradb/storage/langchain_astradb.storage.AstraDBByteStore.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)