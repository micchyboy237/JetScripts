from jet.logger import logger
from langchain.storage import LocalFileStore
from pathlib import Path
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
sidebar_label: Local Filesystem
---

# LocalFileStore

This will help you get started with local filesystem [key-value stores](/docs/concepts/key_value_stores). For detailed documentation of all LocalFileStore features and configurations head to the [API reference](https://python.langchain.com/api_reference/langchain/storage/langchain.storage.file_system.LocalFileStore.html).

## Overview

The `LocalFileStore` is a persistent implementation of `ByteStore` that stores everything in a folder of your choosing. It's useful if you're using a single machine and are tolerant of files being added or deleted.

### Integration details

| Class | Package | Local | [JS support](https://js.langchain.com/docs/integrations/stores/file_system) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: |
| [LocalFileStore](https://python.langchain.com/api_reference/langchain/storage/langchain.storage.file_system.LocalFileStore.html) | [langchain](https://python.langchain.com/api_reference/langchain/index.html) | ✅ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain?style=flat-square&label=%20) |

### Installation

The LangChain `LocalFileStore` integration lives in the `langchain` package:
"""
logger.info("# LocalFileStore")

# %pip install -qU langchain

"""
## Instantiation

Now we can instantiate our byte store:
"""
logger.info("## Instantiation")



root_path = Path.cwd() / "data"  # can also be a path set by a string

kv_store = LocalFileStore(root_path)

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
You can see the created files in your `data` folder:
"""
logger.info("You can see the created files in your `data` folder:")

# !ls {root_path}

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

For detailed documentation of all `LocalFileStore` features and configurations, head to the API reference: https://python.langchain.com/api_reference/langchain/storage/langchain.storage.file_system.LocalFileStore.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)