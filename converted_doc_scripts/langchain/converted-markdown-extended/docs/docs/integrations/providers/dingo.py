from jet.logger import logger
from langchain_community.vectorstores import Dingo
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
# DingoDB

>[DingoDB](https://github.com/dingodb) is a distributed multi-modal vector
> database. It combines the features of a data lake and a vector database,
> allowing for the storage of any type of data (key-value, PDF, audio,
> video, etc.) regardless of its size. Utilizing DingoDB, you can construct
> your own Vector Ocean (the next-generation data architecture following data
> warehouse and data lake). This enables
> the analysis of both structured and unstructured data through
> a singular SQL with exceptionally low latency in real time.

## Installation and Setup

Install the Python SDK
"""
logger.info("# DingoDB")

pip install dingodb

"""
## VectorStore

There exists a wrapper around DingoDB indexes, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
"""
logger.info("## VectorStore")


"""
For a more detailed walkthrough of the DingoDB wrapper, see [this notebook](/docs/integrations/vectorstores/dingo)
"""
logger.info("For a more detailed walkthrough of the DingoDB wrapper, see [this notebook](/docs/integrations/vectorstores/dingo)")

logger.info("\n\n[DONE]", bright=True)