from jet.logger import logger
from langchain_community.vectorstores import Epsilla
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
# Epsilla

This page covers how to use [Epsilla](https://github.com/epsilla-cloud/vectordb) within LangChain.
It is broken into two parts: installation and setup, and then references to specific Epsilla wrappers.

## Installation and Setup

- Install the Python SDK with `pip/pip3 install pyepsilla`

## Wrappers

### VectorStore

There exists a wrapper around Epsilla vector databases, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
"""
logger.info("# Epsilla")


"""
For a more detailed walkthrough of the Epsilla wrapper, see [this notebook](/docs/integrations/vectorstores/epsilla)
"""
logger.info("For a more detailed walkthrough of the Epsilla wrapper, see [this notebook](/docs/integrations/vectorstores/epsilla)")

logger.info("\n\n[DONE]", bright=True)