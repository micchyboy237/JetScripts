from jet.logger import logger
from langchain_community.vectorstores.pgvector import PGVector
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
# PGVector

This page covers how to use the Postgres [PGVector](https://github.com/pgvector/pgvector) ecosystem within LangChain
It is broken into two parts: installation and setup, and then references to specific PGVector wrappers.

## Installation
- Install the Python package with `pip install pgvector`


## Setup
1. The first step is to create a database with the `pgvector` extension installed.

    Follow the steps at [PGVector Installation Steps](https://github.com/pgvector/pgvector#installation) to install the database and the extension. The docker image is the easiest way to get started.

## Wrappers

### VectorStore

There exists a wrapper around Postgres vector databases, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
"""
logger.info("# PGVector")


"""
### Usage

For a more detailed walkthrough of the PGVector Wrapper, see [this notebook](/docs/integrations/vectorstores/pgvector)
"""
logger.info("### Usage")

logger.info("\n\n[DONE]", bright=True)