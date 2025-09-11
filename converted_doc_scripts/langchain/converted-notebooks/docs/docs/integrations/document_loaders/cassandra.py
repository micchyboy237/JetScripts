from cassandra.cluster import Cluster
from jet.logger import logger
from langchain_community.document_loaders import CassandraLoader
import cassio
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

[Cassandra](https://cassandra.apache.org/) is a NoSQL, row-oriented, highly scalable and highly available database.Starting with version 5.0, the database ships with [vector search capabilities](https://cassandra.apache.org/doc/trunk/cassandra/vector-search/overview.html).

## Overview

The Cassandra Document Loader returns a list of Langchain Documents from a Cassandra database.

You must either provide a CQL query or a table name to retrieve the documents.
The Loader takes the following parameters:

* table: (Optional) The table to load the data from.
* session: (Optional) The cassandra driver session. If not provided, the cassio resolved session will be used.
* keyspace: (Optional) The keyspace of the table. If not provided, the cassio resolved keyspace will be used.
* query: (Optional) The query used to load the data.
* page_content_mapper: (Optional) a function to convert a row to string page content. The default converts the row to JSON.
* metadata_mapper: (Optional) a function to convert a row to metadata dict.
* query_parameters: (Optional) The query parameters used when calling session.execute .
* query_timeout: (Optional) The query timeout used when calling session.execute .
* query_custom_payload: (Optional) The query custom_payload used when calling `session.execute`.
* query_execution_profile: (Optional) The query execution_profile used when calling `session.execute`.
* query_host: (Optional) The query host used when calling `session.execute`.
* query_execute_as: (Optional) The query execute_as used when calling `session.execute`.

## Load documents with the Document Loader
"""
logger.info("# Cassandra")


"""
### Init from a cassandra driver Session

You need to create a `cassandra.cluster.Session` object, as described in the [Cassandra driver documentation](https://docs.datastax.com/en/developer/python-driver/latest/api/cassandra/cluster/#module-cassandra.cluster). The details vary (e.g. with network settings and authentication), but this might be something like:
"""
logger.info("### Init from a cassandra driver Session")


cluster = Cluster()
session = cluster.connect()

"""
You need to provide the name of an existing keyspace of the Cassandra instance:
"""
logger.info("You need to provide the name of an existing keyspace of the Cassandra instance:")

CASSANDRA_KEYSPACE = input("CASSANDRA_KEYSPACE = ")

"""
Creating the document loader:
"""
logger.info("Creating the document loader:")

loader = CassandraLoader(
    table="movie_reviews",
    session=session,
    keyspace=CASSANDRA_KEYSPACE,
)

docs = loader.load()

docs[0]

"""
### Init from cassio

It's also possible to use cassio to configure the session and keyspace.
"""
logger.info("### Init from cassio")


cassio.init(contact_points="127.0.0.1", keyspace=CASSANDRA_KEYSPACE)

loader = CassandraLoader(
    table="movie_reviews",
)

docs = loader.load()

"""
#### Attribution statement

> Apache Cassandra, Cassandra and Apache are either registered trademarks or trademarks of the [Apache Software Foundation](http://www.apache.org/) in the United States and/or other countries.
"""
logger.info("#### Attribution statement")

logger.info("\n\n[DONE]", bright=True)