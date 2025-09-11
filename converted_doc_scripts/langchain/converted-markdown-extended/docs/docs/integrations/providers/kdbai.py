from jet.logger import logger
from langchain_community.vectorstores import KDBAI
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
# KDB.AI

>[KDB.AI](https://kdb.ai) is a powerful knowledge-based vector database and search engine that allows you to build scalable, reliable AI applications, using real-time data, by providing advanced search, recommendation and personalization.


## Installation and Setup

Install the Python SDK:
"""
logger.info("# KDB.AI")

pip install kdbai-client

"""
## Vector store

There exists a wrapper around KDB.AI indexes, allowing you to use it as a vectorstore,
whether for semantic search or example selection.
"""
logger.info("## Vector store")


"""
For a more detailed walkthrough of the KDB.AI vectorstore, see [this notebook](/docs/integrations/vectorstores/kdbai)
"""
logger.info("For a more detailed walkthrough of the KDB.AI vectorstore, see [this notebook](/docs/integrations/vectorstores/kdbai)")

logger.info("\n\n[DONE]", bright=True)