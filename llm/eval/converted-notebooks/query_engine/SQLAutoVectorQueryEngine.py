"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/SQLAutoVectorQueryEngine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# SQL Auto Vector Query Engine
In this tutorial, we show you how to use our SQLAutoVectorQueryEngine.

This query engine allows you to combine insights from your structured tables with your unstructured data.
It first decides whether to query your structured tables for insights.
Once it does, it can then infer a corresponding query to the vector store in order to fetch corresponding documents.

**NOTE:** Any Text-to-SQL application should be aware that executing 
arbitrary SQL queries can be a security risk. It is recommended to
take precautions as needed, such as using restricted roles, read-only
databases, sandboxing, etc.
"""

# %pip install llama-index-vector-stores-pinecone
# %pip install llama-index-readers-wikipedia
# %pip install llama-index-llms-ollama

import openai
import os

# os.environ["OPENAI_API_KEY"] = "[You API key]"

"""
### Setup
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# !pip install llama-index

import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
### Create Common Objects

This includes a `ServiceContext` object containing abstractions such as the LLM and chunk size.
This also includes a `StorageContext` object containing our vector store abstractions.
"""

import pinecone
import os

api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="us-west1-gcp-free")

pinecone_index = pinecone.Index("quickstart")

pinecone_index.delete(deleteAll=True)

from llama_index.core import StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex


vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, namespace="wiki_cities"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex([], storage_context=storage_context)

"""
### Create Database Schema + Test Data

Here we introduce a toy scenario where there are 100 tables (too big to fit into the prompt)
"""

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)

engine = create_engine("sqlite:///:memory:", future=True)
metadata_obj = MetaData()

table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)

metadata_obj.create_all(engine)

metadata_obj.tables.keys()

"""
We introduce some test data into the `city_stats` table
"""

from sqlalchemy import insert

rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

with engine.connect() as connection:
    cursor = connection.exec_driver_sql("SELECT * FROM city_stats")
    print(cursor.fetchall())

"""
### Load Data

We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.
"""

# !pip install wikipedia

from llama_index.readers.wikipedia import WikipediaReader

cities = ["Toronto", "Berlin", "Tokyo"]
wiki_docs = WikipediaReader().load_data(pages=cities)

"""
### Build SQL Index
"""

from llama_index.core import SQLDatabase

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

from llama_index.core.query_engine import NLSQLTableQueryEngine

sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"],
)

"""
### Build Vector Index
"""

from llama_index.core import Settings

for city, wiki_doc in zip(cities, wiki_docs):
    nodes = Settings.node_parser.get_nodes_from_documents([wiki_doc])
    for node in nodes:
        node.metadata = {"title": city}
    vector_index.insert_nodes(nodes)

"""
### Define Query Engines, Set as Tools
"""

from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import RetrieverQueryEngine


vector_store_info = VectorStoreInfo(
    content_info="articles about different cities",
    metadata_info=[
        MetadataInfo(
            name="title", type="str", description="The name of the city"
        ),
    ],
)
vector_auto_retriever = VectorIndexAutoRetriever(
    vector_index, vector_store_info=vector_store_info
)

retriever_query_engine = RetrieverQueryEngine.from_args(
    vector_auto_retriever, llm=Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)
)

from llama_index.core.tools import QueryEngineTool

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        "Useful for translating a natural language query into a SQL query over"
        " a table containing: city_stats, containing the population/country of"
        " each city"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=retriever_query_engine,
    description=(
        f"Useful for answering semantic questions about different cities"
    ),
)

"""
### Define SQLAutoVectorQueryEngine
"""

from llama_index.core.query_engine import SQLAutoVectorQueryEngine

query_engine = SQLAutoVectorQueryEngine(
    sql_tool, vector_tool, llm=Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)
)

response = query_engine.query(
    "Tell me about the arts and culture of the city with the highest"
    " population"
)

print(str(response))

response = query_engine.query("Tell me about the history of Berlin")

print(str(response))

response = query_engine.query(
    "Can you give me the country corresponding to each city?"
)

print(str(response))