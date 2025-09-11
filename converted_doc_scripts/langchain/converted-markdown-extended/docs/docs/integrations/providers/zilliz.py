from jet.logger import logger
from langchain_community.vectorstores import Milvus
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
# Zilliz

>[Zilliz Cloud](https://zilliz.com/doc/quick_start) is a fully managed service on cloud for `LF AI Milvus®`,


## Installation and Setup

Install the Python SDK:
"""
logger.info("# Zilliz")

pip install pymilvus

"""
## Vectorstore

A wrapper around Zilliz indexes allows you to use it as a vectorstore,
whether for semantic search or example selection.
"""
logger.info("## Vectorstore")


"""
For a more detailed walkthrough of the Miluvs wrapper, see [this notebook](/docs/integrations/vectorstores/zilliz)
"""
logger.info("For a more detailed walkthrough of the Miluvs wrapper, see [this notebook](/docs/integrations/vectorstores/zilliz)")

logger.info("\n\n[DONE]", bright=True)