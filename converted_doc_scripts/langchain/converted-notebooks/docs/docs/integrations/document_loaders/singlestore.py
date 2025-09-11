from jet.logger import logger
from langchain_singlestore.document_loaders import SingleStoreLoader
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
sidebar_label: SingleStore
---

# SingleStoreLoader

The `SingleStoreLoader` allows you to load documents directly from a SingleStore database table. It is part of the `langchain-singlestore` integration package.

## Overview

### Integration Details

| Class | Package | JS Support |
| :--- | :--- | :---: |
| `SingleStoreLoader` | `langchain_singlestore` | ❌ |

### Features
- Load documents lazily to handle large datasets efficiently.
- Supports native asynchronous operations.
- Easily configurable to work with different database schemas.

## Setup

To use the `SingleStoreLoader`, you need to install the `langchain-singlestore` package. Follow the installation instructions below.

### Installation

Install **langchain_singlestore**.
"""
logger.info("# SingleStoreLoader")

# %pip install -qU langchain_singlestore

"""
## Initialization

To initialize `SingleStoreLoader`, you need to provide connection parameters for the SingleStore database and specify the table and fields to load documents from.

### Required Parameters:
- **host** (`str`): Hostname, IP address, or URL for the database.
- **table_name** (`str`): Name of the table to query. Defaults to `embeddings`.
- **content_field** (`str`): Field containing document content. Defaults to `content`.
- **metadata_field** (`str`): Field containing document metadata. Defaults to `metadata`.

### Optional Parameters:
- **id_field** (`str`): Field containing document IDs. Defaults to `id`.

### Connection Pool Parameters:
- **pool_size** (`int`): Number of active connections in the pool. Defaults to `5`.
- **max_overflow** (`int`): Maximum connections beyond `pool_size`. Defaults to `10`.
- **timeout** (`float`): Connection timeout in seconds. Defaults to `30`.

### Additional Options:
- **pure_python** (`bool`): Enables pure Python mode.
- **local_infile** (`bool`): Allows local file uploads.
- **charset** (`str`): Character set for string values.
- **ssl_key**, **ssl_cert**, **ssl_ca** (`str`): Paths to SSL files.
- **ssl_disabled** (`bool`): Disables SSL.
- **ssl_verify_cert** (`bool`): Verifies server's certificate.
- **ssl_verify_identity** (`bool`): Verifies server's identity.
- **autocommit** (`bool`): Enables autocommits.
- **results_type** (`str`): Structure of query results (e.g., `tuples`, `dicts`).
"""
logger.info("## Initialization")


loader = SingleStoreLoader(
    host="127.0.0.1:3306/db",
    table_name="documents",
    content_field="content",
    metadata_field="metadata",
    id_field="id",
)

"""
## Load
"""
logger.info("## Load")

docs = loader.load()
docs[0]

logger.debug(docs[0].metadata)

"""
## Lazy Load
"""
logger.info("## Lazy Load")

page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:

        page = []

"""
## API reference

For detailed documentation of all SingleStore Document Loader features and configurations head to the github page: [https://github.com/singlestore-labs/langchain-singlestore/](https://github.com/singlestore-labs/langchain-singlestore/)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)