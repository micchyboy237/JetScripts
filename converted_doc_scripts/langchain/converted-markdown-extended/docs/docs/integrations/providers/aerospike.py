from jet.logger import logger
from langchain_aerospike.vectorstores import Aerospike
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
# Aerospike

>[Aerospike](https://aerospike.com/docs/vector) is a high-performance, distributed database known for its speed and scalability, now with support for vector storage and search, enabling retrieval and search of embedding vectors for machine learning and AI applications.
> See the documentation for Aerospike Vector Search (AVS) [here](https://aerospike.com/docs/vector).

## Installation and Setup

Install the AVS Python SDK and AVS langchain vector store:
"""
logger.info("# Aerospike")

pip install aerospike-vector-search langchain-aerospike

"""
See the documentation for the Python SDK [here](https://aerospike-vector-search-python-client.readthedocs.io/en/latest/index.html).
The documentation for the AVS langchain vector store is [here](https://langchain-aerospike.readthedocs.io/en/latest/).

## Vector Store

To import this vectorstore:
"""
logger.info("## Vector Store")


"""
See a usage example [here](https://python.langchain.com/docs/integrations/vectorstores/aerospike/).
"""
logger.info("See a usage example [here](https://python.langchain.com/docs/integrations/vectorstores/aerospike/).")

logger.info("\n\n[DONE]", bright=True)