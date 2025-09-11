from __module_name__ import __ModuleName__ByteStore
from jet.logger import logger
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
sidebar_label: __ModuleName__ByteStore
---

# __ModuleName__ByteStore

- TODO: Make sure API reference link is correct.

This will help you get started with __ModuleName__ [key-value stores](/docs/concepts/#key-value-stores). For detailed documentation of all __ModuleName__ByteStore features and configurations head to the [API reference](https://python.langchain.com/v0.2/api_reference/core/stores/langchain_core.stores.__module_name__ByteStore.html).

- TODO: Add any other relevant links, like information about models, prices, context windows, etc. See https://python.langchain.com/docs/integrations/stores/in_memory/ for an example.

## Overview

- TODO: (Optional) A short introduction to the underlying technology/API.

### Integration details

- TODO: Fill in table features.
- TODO: Remove JS support link if not relevant, otherwise ensure link is correct.
- TODO: Make sure API reference links are correct.

| Class | Package | Local | [JS support](https://js.langchain.com/docs/integrations/stores/_package_name_) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: |
| [__ModuleName__ByteStore](https://api.python.langchain.com/en/latest/stores/__module_name__.stores.__ModuleName__ByteStore.html) | [__package_name__](https://api.python.langchain.com/en/latest/__package_name_short_snake___api_reference.html) | ✅/❌ | ✅/❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/__package_name__?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/__package_name__?style=flat-square&label=%20) |

## Setup

- TODO: Update with relevant info.

To create a __ModuleName__ byte store, you'll need to create a/an __ModuleName__ account, get an API key, and install the `__package_name__` integration package.

### Credentials

- TODO: Update with relevant info, or omit if the service does not require any credentials.

Head to (TODO: link) to sign up to __ModuleName__ and generate an API key. Once you've done this set the __MODULE_NAME___API_KEY environment variable:
"""
logger.info("# __ModuleName__ByteStore")

# import getpass

if not os.getenv("__MODULE_NAME___API_KEY"):
#     os.environ["__MODULE_NAME___API_KEY"] = getpass.getpass("Enter your __ModuleName__ API key: ")

"""
### Installation

The LangChain __ModuleName__ integration lives in the `__package_name__` package:
"""
logger.info("### Installation")

# %pip install -qU __package_name__

"""
## Instantiation

Now we can instantiate our byte store:

- TODO: Update model instantiation with relevant params.
"""
logger.info("## Instantiation")


kv_store = __ModuleName__ByteStore(
)

"""
## Usage

- TODO: Run cells so output can be seen.

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
## TODO: Any functionality specific to this key-value store provider

E.g. extra initialization. Delete if not relevant.

## API reference

For detailed documentation of all __ModuleName__ByteStore features and configurations, head to the API reference: https://api.python.langchain.com/en/latest/stores/__module_name__.stores.__ModuleName__ByteStore.html
"""
logger.info("## TODO: Any functionality specific to this key-value store provider")

logger.info("\n\n[DONE]", bright=True)