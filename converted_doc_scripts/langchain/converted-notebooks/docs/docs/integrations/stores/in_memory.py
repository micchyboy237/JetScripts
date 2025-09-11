from jet.logger import logger
from langchain_core.stores import InMemoryByteStore
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
sidebar_label: In-memory
---

# InMemoryByteStore

This guide will help you get started with in-memory [key-value stores](/docs/concepts/key_value_stores). For detailed documentation of all `InMemoryByteStore` features and configurations head to the [API reference](https://python.langchain.com/api_reference/core/stores/langchain_core.stores.InMemoryByteStore.html).

## Overview

The `InMemoryByteStore` is a non-persistent implementation of a `ByteStore` that stores everything in a Python dictionary. It's intended for demos and cases where you don't need persistence past the lifetime of the Python process.

### Integration details

| Class | Package | Local | [JS support](https://js.langchain.com/docs/integrations/stores/in_memory/) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: |
| [InMemoryByteStore](https://python.langchain.com/api_reference/core/stores/langchain_core.stores.InMemoryByteStore.html) | [langchain-core](https://python.langchain.com/api_reference/core/index.html) | ✅ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_core?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_core?style=flat-square&label=%20) |

### Installation

The LangChain `InMemoryByteStore` integration lives in the `langchain-core` package:
"""
logger.info("# InMemoryByteStore")

# %pip install -qU langchain-core

"""
## Instantiation

Now you can instantiate your byte store:
"""
logger.info("## Instantiation")


kv_store = InMemoryByteStore()

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

For detailed documentation of all `InMemoryByteStore` features and configurations, head to the API reference: https://python.langchain.com/api_reference/core/stores/langchain_core.stores.InMemoryByteStore.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)