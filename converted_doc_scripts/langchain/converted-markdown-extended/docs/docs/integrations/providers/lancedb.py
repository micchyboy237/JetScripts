from jet.logger import logger
from langchain_community.vectorstores import LanceDB
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
# LanceDB

This page covers how to use [LanceDB](https://github.com/lancedb/lancedb) within LangChain.
It is broken into two parts: installation and setup, and then references to specific LanceDB wrappers.

## Installation and Setup

- Install the Python SDK with `pip install lancedb`

## Wrappers

### VectorStore

There exists a wrapper around LanceDB databases, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
"""
logger.info("# LanceDB")


"""
For a more detailed walkthrough of the LanceDB wrapper, see [this notebook](/docs/integrations/vectorstores/lancedb)
"""
logger.info("For a more detailed walkthrough of the LanceDB wrapper, see [this notebook](/docs/integrations/vectorstores/lancedb)")

logger.info("\n\n[DONE]", bright=True)