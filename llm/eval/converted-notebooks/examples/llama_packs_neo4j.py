from script_utils import display_source_nodes
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Llama Pack - Neo4j Query Engine
# 
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-neo4j-query-engine/examples/llama_packs_neo4j.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# This Llama Pack creates a Neo4j knowledge graph query engine, and executes its `query` function. This pack offers the option of creating multiple types of query engines for Neo4j knowledge graphs, namely:
# 
# * Knowledge graph vector-based entity retrieval (default if no query engine type option is provided)
# * Knowledge graph keyword-based entity retrieval
# * Knowledge graph hybrid entity retrieval
# * Raw vector index retrieval
# * Custom combo query engine (vector similarity + KG entity retrieval)
# * KnowledgeGraphQueryEngine
# * KnowledgeGraphRAGRetriever
# 
# For this notebook, we will load a Wikipedia page on paleo diet into Neo4j KG and perform queries.

# %pip install llama-index-readers-wikipedia
# %pip install llama-hub-llama-packs-neo4j-query-engine-base

# !pip install llama_index llama_hub neo4j

import os, openai, logging, sys

# os.environ["OPENAI_API_KEY"] = "sk-#######################"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

## Setup Data
# 
# Load a Wikipedia page on paleo diet.

from llama_index.readers.wikipedia import WikipediaReader

loader = WikipediaReader()
documents = loader.load_data(pages=["Paleolithic diet"], auto_suggest=False)
print(f"Loaded {len(documents)} documents")

## Download and Initialize Pack

from llama_index.core.llama_pack import download_llama_pack

Neo4jQueryEnginePack = download_llama_pack("Neo4jQueryEnginePack", "./neo4j_pack")

# Assume you have the credentials for Neo4j stored in `credentials.json` at the project root, you load the json and extract the credential details.

import json

with open("credentials.json") as f:
    neo4j_connection_params = json.load(f)
    username = neo4j_connection_params["username"]
    password = neo4j_connection_params["password"]
    url = neo4j_connection_params["url"]
    database = neo4j_connection_params["database"]

# See below how `Neo4jQueryEnginePack` is constructed.  You can pass in the `query_engine_type` from `Neo4jQueryEngineType` to construct `Neo4jQueryEnginePack`. The code snippet below shows a KG keyword query engine.  If `query_engine_type` is not defined, it defaults to KG vector based entity retrieval.
# 
# `Neo4jQueryEngineType` is an enum, which holds various query engine types, see below. You can pass in any of these query engine types to construct `Neo4jQueryEnginePack`.
# ```
# class Neo4jQueryEngineType(str, Enum):
#     """Neo4j query engine type"""
# 
#     KG_KEYWORD = "keyword"
#     KG_HYBRID = "hybrid"
#     RAW_VECTOR = "vector"
#     RAW_VECTOR_KG_COMBO = "vector_kg"
#     KG_QE = "KnowledgeGraphQueryEngine"
#     KG_RAG_RETRIEVER = "KnowledgeGraphRAGRetriever"
# ```

from llama_index.packs.neo4j_query_engine.base import Neo4jQueryEngineType

neo4j_pack = Neo4jQueryEnginePack(
    username=username,
    password=password,
    url=url,
    database=database,
    docs=documents,
    query_engine_type=Neo4jQueryEngineType.KG_KEYWORD,
)

## Run Pack

query = "Tell me about the benefits of paleo diet."

response = neo4j_pack.run(query)
logger.newline()
logger.info("Query KG_KEYWORD:")
display_source_nodes(query, response)

# Let's try out the KG hybrid query engine. See code below.  You can try any other query engines in a similar way by replacing the `query_engine_type` with another query engine type from `Neo4jQueryEngineType` enum.

neo4j_pack = Neo4jQueryEnginePack(
    username=username,
    password=password,
    url=url,
    database=database,
    docs=documents,
    query_engine_type=Neo4jQueryEngineType.KG_HYBRID,
)


response = neo4j_pack.run(query)
logger.newline()
logger.info("Query KG_HYBRID:")
display_source_nodes(query, response)

## Comparison of the Knowledge Graph Query Strategies
# 
# The table below lists the details of the 7 query engines, and their pros and cons based on experiments with NebulaGraph and LlamaIndex, as outlined in the blog post [7 Query Strategies for Navigating Knowledge Graphs with LlamaIndex](https://betterprogramming.pub/7-query-strategies-for-navigating-knowledge-graphs-with-llamaindex-ed551863d416?sk=55c94ad72e75aa52ac6cc21d8145b37d).

# ![Knowledge Graph query strategies comparison](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*0UsLpj7v2GO67U-99YJBfg.png)

logger.info("\n\n[DONE]", bright=True)