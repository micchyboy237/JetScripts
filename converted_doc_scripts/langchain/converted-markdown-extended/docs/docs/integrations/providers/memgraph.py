from jet.logger import logger
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_memgraph import MemgraphToolkit
from langchain_memgraph.chains.graph_qa import MemgraphQAChain
from langchain_memgraph.graphs.memgraph import MemgraphLangChain
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
# Memgraph

>Memgraph is a high-performance, in-memory graph database that is optimized for real-time queries and analytics.
>Get started with Memgraph by visiting [their website](https://memgraph.com/).

## Installation and Setup

- Install the Python SDK with `pip install langchain-memgraph`

## MemgraphQAChain

There exists a wrapper around Memgraph graph database that allows you to generate Cypher statements based on the user input
and use them to retrieve relevant information from the database.
"""
logger.info("# Memgraph")


"""
See a [usage example](/docs/integrations/graphs/memgraph)

## Constructing a Knowledge Graph from unstructured data

You can use the integration to construct a knowledge graph from unstructured data.
"""
logger.info("## Constructing a Knowledge Graph from unstructured data")


"""
See a [usage example](/docs/integrations/graphs/memgraph)

## Memgraph Tools and Toolkit

Memgraph also provides a toolkit that allows you to interact with the Memgraph database.
See a [usage example](/docs/integrations/tools/memgraph).
"""
logger.info("## Memgraph Tools and Toolkit")


logger.info("\n\n[DONE]", bright=True)