from cassandra.cluster import Cluster
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
Document,
StorageContext,
)
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.cassandra import CassandraVectorStore
import cassio
import os
import shutil


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/CassandraIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Cassandra Vector Store

[Apache Cassandra®](https://cassandra.apache.org) is a NoSQL, row-oriented, highly scalable and highly available database. Starting with version 5.0, the database ships with [vector search](https://cassandra.apache.org/doc/trunk/cassandra/vector-search/overview.html) capabilities.

DataStax [Astra DB through CQL](https://docs.datastax.com/en/astra-serverless/docs/vector-search/quickstart.html) is a managed serverless database built on Cassandra, offering the same interface and strengths.

**This notebook shows the basic usage of the Cassandra Vector Store in LlamaIndex.**

To run the full code you need either a running Cassandra cluster equipped with Vector 
Search capabilities or a DataStax Astra DB instance.

## Setup
"""
logger.info("# Cassandra Vector Store")

# %pip install llama-index-vector-stores-cassandra

# !pip install --quiet "astrapy>=0.5.8"

# from getpass import getpass


"""
The next step is to initialize CassIO with a global DB connection: this is the only step that is done slightly differently for a Cassandra cluster and Astra DB:

### Initialization (Cassandra cluster)

In this case, you first need to create a `cassandra.cluster.Session` object,
as described in the [Cassandra driver documentation](https://docs.datastax.com/en/developer/python-driver/latest/api/cassandra/cluster/#module-cassandra.cluster).
The details vary (e.g. with network settings and authentication), but this might be something like:
"""
logger.info("### Initialization (Cassandra cluster)")


cluster = Cluster(["127.0.0.1"])
session = cluster.connect()


CASSANDRA_KEYSPACE = input("CASSANDRA_KEYSPACE = ")

cassio.init(session=session, keyspace=CASSANDRA_KEYSPACE)

"""
### Initialization (Astra DB through CQL)

In this case you initialize CassIO with the following connection parameters:

- the Database ID, e.g. 01234567-89ab-cdef-0123-456789abcdef
- the Token, e.g. AstraCS:6gBhNmsk135.... (it must be a "Database Administrator" token)
- Optionally a Keyspace name (if omitted, the default one for the database will be used)
"""
logger.info("### Initialization (Astra DB through CQL)")

ASTRA_DB_ID = input("ASTRA_DB_ID = ")
# ASTRA_DB_TOKEN = getpass("ASTRA_DB_TOKEN = ")

desired_keyspace = input("ASTRA_DB_KEYSPACE (optional, can be left empty) = ")
if desired_keyspace:
    ASTRA_DB_KEYSPACE = desired_keyspace
else:
    ASTRA_DB_KEYSPACE = None


cassio.init(
    database_id=ASTRA_DB_ID,
    token=ASTRA_DB_TOKEN,
    keyspace=ASTRA_DB_KEYSPACE,
)

"""
### MLX key

In order to use embeddings by MLX you need to supply an MLX API Key:
"""
logger.info("### MLX key")

# os.environ["OPENAI_API_KEY"] = getpass("MLX API Key:")

"""
### Download data
"""
logger.info("### Download data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Creating and populating the Vector Store

You will now load some essays by Paul Graham from a local file and store them into the Cassandra Vector Store.
"""
logger.info("## Creating and populating the Vector Store")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
logger.debug(f"Total documents: {len(documents)}")
logger.debug(f"First document, id: {documents[0].doc_id}")
logger.debug(f"First document, hash: {documents[0].hash}")
logger.debug(
    "First document, text"
    f" ({len(documents[0].text)} characters):\n{'='*20}\n{documents[0].text[:360]} ..."
)

"""
### Initialize the Cassandra Vector Store

Creation of the vector store entails creation of the underlying database table if it does not exist yet:
"""
logger.info("### Initialize the Cassandra Vector Store")

cassandra_store = CassandraVectorStore(
    table="cass_v_table", embedding_dimension=1536
)

"""
Now wrap this store into an `index` LlamaIndex abstraction for later querying:
"""
logger.info("Now wrap this store into an `index` LlamaIndex abstraction for later querying:")

storage_context = StorageContext.from_defaults(vector_store=cassandra_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
Note that the above `from_documents` call does several things at once: it splits the input documents into chunks of manageable size ("nodes"), computes embedding vectors for each node, and stores them all in the Cassandra Vector Store.

## Querying the store

### Basic querying
"""
logger.info("## Querying the store")

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")
logger.debug(response.response)

"""
### MMR-based queries

The MMR (maximal marginal relevance) method is designed to fetch text chunks from the store that are at the same time relevant to the query but as different as possible from each other, with the goal of providing a broader context to the building of the final answer:
"""
logger.info("### MMR-based queries")

query_engine = index.as_query_engine(vector_store_query_mode="mmr")
response = query_engine.query("Why did the author choose to work on AI?")
logger.debug(response.response)

"""
## Connecting to an existing store

Since this store is backed by Cassandra, it is persistent by definition. So, if you want to connect to a store that was created and populated previously, here is how:
"""
logger.info("## Connecting to an existing store")

new_store_instance = CassandraVectorStore(
    table="cass_v_table", embedding_dimension=1536
)

new_index_instance = VectorStoreIndex.from_vector_store(
    vector_store=new_store_instance
)

query_engine = new_index_instance.as_query_engine(similarity_top_k=5)
response = query_engine.query(
    "What did the author study prior to working on AI?"
)

logger.debug(response.response)

"""
## Removing documents from the index

First get an explicit list of pieces of a document, or "nodes", from a `Retriever` spawned from the index:
"""
logger.info("## Removing documents from the index")

retriever = new_index_instance.as_retriever(
    vector_store_query_mode="mmr",
    similarity_top_k=3,
    vector_store_kwargs={"mmr_prefetch_factor": 4},
)
nodes_with_scores = retriever.retrieve(
    "What did the author study prior to working on AI?"
)

logger.debug(f"Found {len(nodes_with_scores)} nodes.")
for idx, node_with_score in enumerate(nodes_with_scores):
    logger.debug(f"    [{idx}] score = {node_with_score.score}")
    logger.debug(f"        id    = {node_with_score.node.node_id}")
    logger.debug(f"        text  = {node_with_score.node.text[:90]} ...")

"""
But wait! When using the vector store, you should consider the **document** as the sensible unit to delete, and not any individual node belonging to it. Well, in this case, you just inserted a single text file, so all nodes will have the same `ref_doc_id`:
"""
logger.info("But wait! When using the vector store, you should consider the **document** as the sensible unit to delete, and not any individual node belonging to it. Well, in this case, you just inserted a single text file, so all nodes will have the same `ref_doc_id`:")

logger.debug("Nodes' ref_doc_id:")
logger.debug("\n".join([nws.node.ref_doc_id for nws in nodes_with_scores]))

"""
Now let's say you need to remove the text file you uploaded:
"""
logger.info("Now let's say you need to remove the text file you uploaded:")

new_store_instance.delete(nodes_with_scores[0].node.ref_doc_id)

"""
Repeat the very same query and check the results now. You should see _no results_ being found:
"""
logger.info("Repeat the very same query and check the results now. You should see _no results_ being found:")

nodes_with_scores = retriever.retrieve(
    "What did the author study prior to working on AI?"
)

logger.debug(f"Found {len(nodes_with_scores)} nodes.")

"""
## Metadata filtering

The Cassandra vector store support metadata filtering in the form of exact-match `key=value` pairs at query time. The following cells, which work on a brand new Cassandra table, demonstrate this feature.

In this demo, for the sake of brevity, a single source document is loaded (the `./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/paul_graham_essay.txt` text file). Nevertheless, you will attach some custom metadata to the document to illustrate how you can can restrict queries with conditions on the metadata attached to the documents.
"""
logger.info("## Metadata filtering")

md_storage_context = StorageContext.from_defaults(
    vector_store=CassandraVectorStore(
        table="cass_v_table_md", embedding_dimension=1536
    )
)


def my_file_metadata(file_name: str):
    """Depending on the input file name, associate a different metadata."""
    if "essay" in file_name:
        source_type = "essay"
    elif "dinosaur" in file_name:
        source_type = "dinos"
    else:
        source_type = "other"
    return {"source_type": source_type}


md_documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data", file_metadata=my_file_metadata
).load_data()
md_index = VectorStoreIndex.from_documents(
    md_documents, storage_context=md_storage_context
)

"""
That's it: you can now add filtering to your query engine:
"""
logger.info("That's it: you can now add filtering to your query engine:")


md_query_engine = md_index.as_query_engine(
    filters=MetadataFilters(
        filters=[ExactMatchFilter(key="source_type", value="essay")]
    )
)
md_response = md_query_engine.query(
    "did the author appreciate Lisp and painting?"
)
logger.debug(md_response.response)

"""
To test that the filtering is at play, try to change it to use only `"dinos"` documents... there will be no answer this time :)
"""
logger.info("To test that the filtering is at play, try to change it to use only `"dinos"` documents... there will be no answer this time :)")

logger.info("\n\n[DONE]", bright=True)