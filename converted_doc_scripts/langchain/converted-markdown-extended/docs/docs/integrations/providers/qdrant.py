from jet.logger import logger
from langchain_qdrant import FastEmbedSparse
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import SparseEmbeddings
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
# Qdrant

>[Qdrant](https://qdrant.tech/documentation/) (read: quadrant) is a vector similarity search engine.
> It provides a production-ready service with a convenient API to store, search, and manage
> points - vectors with an additional payload. `Qdrant` is tailored to extended filtering support.


## Installation and Setup

Install the Python partner package:
"""
logger.info("# Qdrant")

pip install langchain-qdrant

"""
## Embedding models

### FastEmbedSparse
"""
logger.info("## Embedding models")


"""
### SparseEmbeddings
"""
logger.info("### SparseEmbeddings")


"""
## Vector Store

There exists a wrapper around `Qdrant` indexes, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
"""
logger.info("## Vector Store")


"""
For a more detailed walkthrough of the Qdrant wrapper, see [this notebook](/docs/integrations/vectorstores/qdrant)
"""
logger.info("For a more detailed walkthrough of the Qdrant wrapper, see [this notebook](/docs/integrations/vectorstores/qdrant)")

logger.info("\n\n[DONE]", bright=True)