from jet.logger import logger
from langchain.globals import set_llm_cache
from langchain_community.agent_toolkits.cassandra_database.toolkit import (
CassandraDatabaseToolkit,
)
from langchain_community.cache import CassandraCache
from langchain_community.cache import CassandraSemanticCache
from langchain_community.chat_message_histories import CassandraChatMessageHistory
from langchain_community.document_loaders import CassandraLoader
from langchain_community.tools import GetSchemaCassandraDatabaseTool
from langchain_community.tools import GetTableDataCassandraDatabaseTool
from langchain_community.tools import QueryCassandraDatabaseTool
from langchain_community.vectorstores import Cassandra
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
# Cassandra

> [Apache Cassandra®](https://cassandra.apache.org/) is a NoSQL, row-oriented, highly scalable and highly available database.
> Starting with version 5.0, the database ships with [vector search capabilities](https://cassandra.apache.org/doc/trunk/cassandra/vector-search/overview.html).

The integrations outlined in this page can be used with `Cassandra` as well as other CQL-compatible databases,
i.e. those using the `Cassandra Query Language` protocol.


## Installation and Setup

Install the following Python package:
"""
logger.info("# Cassandra")

pip install "cassio>=0.1.6"

"""
## Vector Store
"""
logger.info("## Vector Store")


"""
Learn more in the [example notebook](/docs/integrations/vectorstores/cassandra).

## Chat message history
"""
logger.info("## Chat message history")


"""
Learn more in the [example notebook](/docs/integrations/memory/cassandra_chat_message_history).


## LLM Cache
"""
logger.info("## LLM Cache")

set_llm_cache(CassandraCache())

"""
Learn more in the [example notebook](/docs/integrations/llm_caching#cassandra-caches) (scroll to the Cassandra section).


## Semantic LLM Cache
"""
logger.info("## Semantic LLM Cache")

set_llm_cache(CassandraSemanticCache(
    embedding=my_embedding,
    table_name="my_store",
))

"""
Learn more in the [example notebook](/docs/integrations/llm_caching#cassandra-caches) (scroll to the appropriate section).

## Document loader
"""
logger.info("## Document loader")


"""
Learn more in the [example notebook](/docs/integrations/document_loaders/cassandra).

#### Attribution statement

> Apache Cassandra, Cassandra and Apache are either registered trademarks or trademarks of
> the [Apache Software Foundation](http://www.apache.org/) in the United States and/or other countries.

## Toolkit

The `Cassandra Database toolkit` enables AI engineers to efficiently integrate agents
with Cassandra data.
"""
logger.info("#### Attribution statement")


"""
Learn more in the [example notebook](/docs/integrations/tools/cassandra_database).


Cassandra Database individual tools:

### Get Schema

Tool for getting the schema of a keyspace in an Apache Cassandra database.
"""
logger.info("### Get Schema")


"""
### Get Table Data

Tool for getting data from a table in an Apache Cassandra database.
"""
logger.info("### Get Table Data")


"""
### Query

Tool for querying an Apache Cassandra database with provided CQL.
"""
logger.info("### Query")


logger.info("\n\n[DONE]", bright=True)