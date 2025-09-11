from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.document_loaders import WikipediaLoader
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_neo4j import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph
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
# Diffbot

>[Diffbot](https://docs.diffbot.com/docs/getting-started-with-diffbot) is a suite of ML-based products that make it easy to structure web data.
>
>Diffbot's [Natural Language Processing API](https://www.diffbot.com/products/natural-language/) allows for the extraction of entities, relationships, and semantic meaning from unstructured text data.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/integrations/graphs/diffbot.ipynb)

## Use case

Text data often contain rich relationships and insights used for various analytics, recommendation engines, or knowledge management applications.

By coupling `Diffbot's NLP API` with `Neo4j`, a graph database, you can create powerful, dynamic graph structures based on the information extracted from text. These graph structures are fully queryable and can be integrated into various applications.

This combination allows for use cases such as:

* Building knowledge graphs (like [Diffbot's Knowledge Graph](https://www.diffbot.com/products/knowledge-graph/)) from textual documents, websites, or social media feeds.
* Generating recommendations based on semantic relationships in the data.
* Creating advanced search features that understand the relationships between entities.
* Building analytics dashboards that allow users to explore the hidden relationships in data.

## Overview

LangChain provides tools to interact with Graph Databases:

1. `Construct knowledge graphs from text` using graph transformer and store integrations 
2. `Query a graph database` using chains for query creation and execution
3. `Interact with a graph database` using agents for robust and flexible querying 

## Setting up

First, get required packages and set environment variables:
"""
logger.info("# Diffbot")

# %pip install --upgrade --quiet  langchain langchain-experimental langchain-ollama langchain-neo4j neo4j wikipedia

"""
### Diffbot NLP API

`Diffbot's NLP API` is a tool for extracting entities, relationships, and semantic context from unstructured text data.
This extracted information can be used to construct a knowledge graph.
To use the API, you'll need to obtain a [free API token from Diffbot](https://app.diffbot.com/get-started/).
"""
logger.info("### Diffbot NLP API")


diffbot_
diffbot_nlp = DiffbotGraphTransformer(diffbot_api_key=diffbot_api_key)

"""
This code fetches Wikipedia articles about "Warren Buffett" and then uses `DiffbotGraphTransformer` to extract entities and relationships.
The `DiffbotGraphTransformer` outputs a structured data `GraphDocument`, which can be used to populate a graph database.
Note that text chunking is avoided due to Diffbot's [character limit per API request](https://docs.diffbot.com/reference/introduction-to-natural-language-api).
"""
logger.info("This code fetches Wikipedia articles about "Warren Buffett" and then uses `DiffbotGraphTransformer` to extract entities and relationships.")


query = "Warren Buffett"
raw_documents = WikipediaLoader(query=query).load()
graph_documents = diffbot_nlp.convert_to_graph_documents(raw_documents)

"""
## Loading the data into a knowledge graph

You will need to have a running Neo4j instance. One option is to create a [free Neo4j database instance in their Aura cloud service](https://neo4j.com/cloud/platform/aura-graph-database/). You can also run the database locally using the [Neo4j Desktop application](https://neo4j.com/download/), or running a docker container. You can run a local docker container by running the executing the following script:
```
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS=\[\"apoc\"\]  \
    neo4j:latest
```    
If you are using the docker container, you need to wait a couple of second for the database to start.
"""
logger.info("## Loading the data into a knowledge graph")


url = "bolt://localhost:7687"
username = "neo4j"
password = "password"

graph = Neo4jGraph(url=url, username=username, password=password)

"""
The `GraphDocuments` can be loaded into a knowledge graph using the `add_graph_documents` method.
"""
logger.info("The `GraphDocuments` can be loaded into a knowledge graph using the `add_graph_documents` method.")

graph.add_graph_documents(graph_documents)

"""
## Refresh graph schema information
If the schema of database changes, you can refresh the schema information needed to generate Cypher statements
"""
logger.info("## Refresh graph schema information")

graph.refresh_schema()

"""
## Querying the graph
We can now use the graph cypher QA chain to ask question of the graph. It is advisable to use **gpt-4** to construct Cypher queries to get the best experience.
"""
logger.info("## Querying the graph")


chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOllama(model="llama3.2"),
    qa_llm=ChatOllama(model="llama3.2"),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

chain.run("Which university did Warren Buffett attend?")

chain.run("Who is or was working at Berkshire Hathaway?")

logger.info("\n\n[DONE]", bright=True)