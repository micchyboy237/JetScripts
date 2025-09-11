from jet.logger import logger
from langchain_community.vectorstores import vlite
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
# vlite

This page covers how to use [vlite](https://github.com/sdan/vlite) within LangChain. vlite is a simple and fast vector database for storing and retrieving embeddings.

## Installation and Setup

To install vlite, run the following command:
"""
logger.info("# vlite")

pip install vlite

"""
For PDF OCR support, install the `vlite[ocr]` extra:
"""
logger.info("For PDF OCR support, install the `vlite[ocr]` extra:")

pip install vlite[ocr]

"""
## VectorStore

vlite provides a wrapper around its vector database, allowing you to use it as a vectorstore for semantic search and example selection.

To import the vlite vectorstore:
"""
logger.info("## VectorStore")


"""
### Usage

For a more detailed walkthrough of the vlite wrapper, see [this notebook](/docs/integrations/vectorstores/vlite).
"""
logger.info("### Usage")

logger.info("\n\n[DONE]", bright=True)