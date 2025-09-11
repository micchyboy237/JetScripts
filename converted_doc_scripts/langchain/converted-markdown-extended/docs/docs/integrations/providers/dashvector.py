from jet.logger import logger
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import DashVector
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
# DashVector

> [DashVector](https://help.aliyun.com/document_detail/2510225.html) is a fully-managed vectorDB service that supports high-dimension dense and sparse vectors, real-time insertion and filtered search. It is built to scale automatically and can adapt to different application requirements.

This document demonstrates to leverage DashVector within the LangChain ecosystem. In particular, it shows how to install DashVector, and how to use it as a VectorStore plugin in LangChain.
It is broken into two parts: installation and setup, and then references to specific DashVector wrappers.

## Installation and Setup


Install the Python SDK:
"""
logger.info("# DashVector")

pip install dashvector

"""
You must have an API key. Here are the [installation instructions](https://help.aliyun.com/document_detail/2510223.html).


## Embedding models
"""
logger.info("## Embedding models")


"""
See the [use example](/docs/integrations/vectorstores/dashvector).


## Vector Store

A DashVector Collection is wrapped as a familiar VectorStore for native usage within LangChain,
which allows it to be readily used for various scenarios, such as semantic search or example selection.

You may import the vectorstore by:
"""
logger.info("## Vector Store")


"""
For a detailed walkthrough of the DashVector wrapper, please refer to [this notebook](/docs/integrations/vectorstores/dashvector)
"""
logger.info("For a detailed walkthrough of the DashVector wrapper, please refer to [this notebook](/docs/integrations/vectorstores/dashvector)")

logger.info("\n\n[DONE]", bright=True)