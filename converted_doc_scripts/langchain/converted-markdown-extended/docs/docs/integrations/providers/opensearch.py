from jet.logger import logger
from langchain_community.vectorstores import OpenSearchVectorSearch
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
# OpenSearch

This page covers how to use the OpenSearch ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific OpenSearch wrappers.

## Installation and Setup
- Install the Python package with `pip install opensearch-py`
## Wrappers

### VectorStore

There exists a wrapper around OpenSearch vector databases, allowing you to use it as a vectorstore
for semantic search using approximate vector search powered by lucene, nmslib and faiss engines
or using painless scripting and script scoring functions for bruteforce vector search.

To import this vectorstore:
"""
logger.info("# OpenSearch")


"""
For a more detailed walkthrough of the OpenSearch wrapper, see [this notebook](/docs/integrations/vectorstores/opensearch)
"""
logger.info("For a more detailed walkthrough of the OpenSearch wrapper, see [this notebook](/docs/integrations/vectorstores/opensearch)")

logger.info("\n\n[DONE]", bright=True)