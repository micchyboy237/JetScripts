from jet.logger import logger
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_neo4j import Neo4jChatMessageHistory
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
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
# Neo4j

>What is `Neo4j`?

>- Neo4j is an `open-source database management system` that specializes in graph database technology.
>- Neo4j allows you to represent and store data in nodes and edges, making it ideal for handling connected data and relationships.
>- Neo4j provides a `Cypher Query Language`, making it easy to interact with and query your graph data.
>- With Neo4j, you can achieve high-performance `graph traversals and queries`, suitable for production-level systems.

>Get started with Neo4j by visiting [their website](https://neo4j.com/).

## Installation and Setup

- Install the Python SDK with `pip install neo4j langchain-neo4j`


## VectorStore

The Neo4j vector index is used as a vectorstore,
whether for semantic search or example selection.
"""
logger.info("# Neo4j")


"""
See a [usage example](/docs/integrations/vectorstores/neo4jvector)

## GraphCypherQAChain

There exists a wrapper around Neo4j graph database that allows you to generate Cypher statements based on the user input
and use them to retrieve relevant information from the database.
"""
logger.info("## GraphCypherQAChain")


"""
See a [usage example](/docs/integrations/graphs/neo4j_cypher)

## Constructing a knowledge graph from text

Text data often contain rich relationships and insights that can be useful for various analytics, recommendation engines, or knowledge management applications.
Diffbot's NLP API allows for the extraction of entities, relationships, and semantic meaning from unstructured text data.
By coupling Diffbot's NLP API with Neo4j, a graph database, you can create powerful, dynamic graph structures based on the information extracted from text.
These graph structures are fully queryable and can be integrated into various applications.
"""
logger.info("## Constructing a knowledge graph from text")


"""
See a [usage example](/docs/integrations/graphs/diffbot)

## Memory

See a [usage example](/docs/integrations/memory/neo4j_chat_message_history).
"""
logger.info("## Memory")


logger.info("\n\n[DONE]", bright=True)