from IPython.display import Markdown, display
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SQLDatabase
from llama_index.core import SQLDatabase, SimpleDirectoryReader, Document
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import (
SQLTableNodeMapping,
ObjectIndex,
SQLTableSchema,
)
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
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
from sqlalchemy import insert
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/index_structs/struct_indices/duckdb_sql_query.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# SQL Query Engine with LlamaIndex + DuckDB

This guide showcases the core LlamaIndex SQL capabilities with DuckDB. 

We go through some core LlamaIndex data structures, including the `NLSQLTableQueryEngine` and `SQLTableRetrieverQueryEngine`. 

**NOTE:** Any Text-to-SQL application should be aware that executing 
arbitrary SQL queries can be a security risk. It is recommended to
take precautions as needed, such as using restricted roles, read-only
databases, sandboxing, etc.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# SQL Query Engine with LlamaIndex + DuckDB")

# %pip install llama-index-readers-wikipedia

# !pip install llama-index

# !pip install duckdb duckdb-engine


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



"""
## Basic Text-to-SQL with our `NLSQLTableQueryEngine` 

In this initial example, we walk through populating a SQL database with some test datapoints, and querying it with our text-to-SQL capabilities.

### Create Database Schema + Test Data

We use sqlalchemy, a popular SQL database toolkit, to connect to DuckDB and create an empty `city_stats` Table. We then populate it with some test data.
"""
logger.info("## Basic Text-to-SQL with our `NLSQLTableQueryEngine`")


engine = create_engine("duckdb:///:memory:")
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
logger.info("We introduce some test data into the `city_stats` table")


rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

with engine.connect() as connection:
    cursor = connection.exec_driver_sql("SELECT * FROM city_stats")
    logger.debug(cursor.fetchall())

"""
### Create SQLDatabase Object

We first define our SQLDatabase abstraction (a light wrapper around SQLAlchemy).
"""
logger.info("### Create SQLDatabase Object")


sql_database = SQLDatabase(engine, include_tables=["city_stats"])

"""
### Query Index

Here we demonstrate the capabilities of `NLSQLTableQueryEngine`, which performs text-to-SQL.

1. We construct a `NLSQLTableQueryEngine` and pass in our SQL database object.
2. We run queries against the query engine.
"""
logger.info("### Query Index")

query_engine = NLSQLTableQueryEngine(sql_database)

response = query_engine.query("Which city has the highest population?")

str(response)

response.metadata

"""
## Advanced Text-to-SQL with our `SQLTableRetrieverQueryEngine` 

In this guide, we tackle the setting where you have a large number of tables in your database, and putting all the table schemas into the prompt may overflow the text-to-SQL prompt.

We first index the schemas with our `ObjectIndex`, and then use our `SQLTableRetrieverQueryEngine` abstraction on top.
"""
logger.info("## Advanced Text-to-SQL with our `SQLTableRetrieverQueryEngine`")

engine = create_engine("duckdb:///:memory:")
metadata_obj = MetaData()

table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
all_table_names = ["city_stats"]
n = 100
for i in range(n):
    tmp_table_name = f"tmp_table_{i}"
    tmp_table = Table(
        tmp_table_name,
        metadata_obj,
        Column(f"tmp_field_{i}_1", String(16), primary_key=True),
        Column(f"tmp_field_{i}_2", Integer),
        Column(f"tmp_field_{i}_3", String(16), nullable=False),
    )
    all_table_names.append(f"tmp_table_{i}")

metadata_obj.create_all(engine)


rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

"""
### Construct Object Index
"""
logger.info("### Construct Object Index")


table_node_mapping = SQLTableNodeMapping(sql_database)

table_schema_objs = []
for table_name in all_table_names:
    table_schema_objs.append(SQLTableSchema(table_name=table_name))

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)

"""
### Query Index with `SQLTableRetrieverQueryEngine`
"""
logger.info("### Query Index with `SQLTableRetrieverQueryEngine`")

query_engine = SQLTableRetrieverQueryEngine(
    sql_database,
    obj_index.as_retriever(similarity_top_k=1),
)

response = query_engine.query("Which city has the highest population?")

response

logger.info("\n\n[DONE]", bright=True)