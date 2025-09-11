from jet.logger import logger
from langchain.retrievers import SelfQueryRetriever
from langchain_chroma import Chroma
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
# Chroma

>[Chroma](https://docs.trychroma.com/getting-started) is a database for building AI applications with embeddings.

## Installation and Setup
"""
logger.info("# Chroma")

pip install langchain-chroma

"""
## VectorStore

There exists a wrapper around Chroma vector databases, allowing you to use it as a vectorstore,
whether for semantic search or example selection.
"""
logger.info("## VectorStore")


"""
For a more detailed walkthrough of the Chroma wrapper, see [this notebook](/docs/integrations/vectorstores/chroma)

## Retriever

See a [usage example](/docs/integrations/retrievers/self_query/chroma_self_query).
"""
logger.info("## Retriever")


logger.info("\n\n[DONE]", bright=True)