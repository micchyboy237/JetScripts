from jet.logger import logger
from langchain_graph_retriever import GraphRetriever
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
# Graph RAG

## Overview

[Graph RAG](https://datastax.github.io/graph-rag/) provides a retriever interface
that combines **unstructured** similarity search on vectors with **structured**
traversal of metadata properties. This enables graph-based retrieval over **existing**
vector stores.

## Installation and setup
"""
logger.info("# Graph RAG")

pip install langchain-graph-retriever

"""
## Retrievers
"""
logger.info("## Retrievers")


"""
For more information, see the [Graph RAG Integration Guide](/docs/integrations/retrievers/graph_rag).
"""
logger.info("For more information, see the [Graph RAG Integration Guide](/docs/integrations/retrievers/graph_rag).")

logger.info("\n\n[DONE]", bright=True)