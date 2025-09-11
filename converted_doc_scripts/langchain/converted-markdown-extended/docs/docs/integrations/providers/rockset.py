from jet.logger import logger
from langchain_community.chat_message_histories import RocksetChatMessageHistory
from langchain_community.document_loaders import RocksetLoader
from langchain_community.vectorstores import Rockset
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
# Rockset

>[Rockset](https://rockset.com/product/) is a real-time analytics database service for serving low latency, high concurrency analytical queries at scale. It builds a Converged Index™ on structured and semi-structured data with an efficient store for vector embeddings. Its support for running SQL on schemaless data makes it a perfect choice for running vector search with metadata filters.

## Installation and Setup

Make sure you have Rockset account and go to the web console to get the API key. Details can be found on [the website](https://rockset.com/docs/rest-api/).
"""
logger.info("# Rockset")

pip install rockset

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/rockset).
"""
logger.info("## Vector Store")


"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/rockset).
"""
logger.info("## Document Loader")


"""
## Chat Message History

See a [usage example](/docs/integrations/memory/rockset_chat_message_history).
"""
logger.info("## Chat Message History")


logger.info("\n\n[DONE]", bright=True)