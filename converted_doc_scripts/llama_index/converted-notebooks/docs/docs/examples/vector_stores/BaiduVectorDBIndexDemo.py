from jet.logger import CustomLogger
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
StorageContext,
)
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.baiduvectordb import (
BaiduVectorDB,
TableParams,
TableField,
)
import openai
import os
import pymochow
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Baidu VectorDB

>[Baidu VectorDB](https://cloud.baidu.com/product/vdb.html) is a robust, enterprise-level distributed database service, meticulously developed and fully managed by Baidu Intelligent Cloud. It stands out for its exceptional ability to store, retrieve, and analyze multi-dimensional vector data. At its core, VectorDB operates on Baidu's proprietary \"Mochow\" vector database kernel, which ensures high performance, availability, and security, alongside remarkable scalability and user-friendliness.

>This database service supports a diverse range of index types and similarity calculation methods, catering to various use cases. A standout feature of VectorDB is its capacity to manage an immense vector scale of up to 10 billion, while maintaining impressive query performance, supporting millions of queries per second (QPS) with millisecond-level query latency.

**This notebook shows the basic usage of BaiduVectorDB as a Vector Store in LlamaIndex.**

To run, you should have a [Database instance.](https://cloud.baidu.com/doc/VDB/s/hlrsoazuf)

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Baidu VectorDB")

# %pip install llama-index-vector-stores-baiduvectordb

# !pip install llama-index

# !pip install pymochow


"""
### Please provide OllamaFunctionCalling access key

In order use embeddings by OllamaFunctionCalling you need to supply an OllamaFunctionCalling API Key:
"""
logger.info("### Please provide OllamaFunctionCalling access key")


# OPENAI_API_KEY = getpass.getpass("OllamaFunctionCalling API Key:")
# openai.api_key = OPENAI_API_KEY

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Creating and populating the Vector Store

You will now load some essays by Paul Graham from a local file and store them into the Baidu VectorDB.
"""
logger.info("## Creating and populating the Vector Store")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
logger.debug(f"Total documents: {len(documents)}")
logger.debug(f"First document, id: {documents[0].doc_id}")
logger.debug(f"First document, hash: {documents[0].hash}")
logger.debug(
    f"First document, text ({len(documents[0].text)} characters):\n{'='*20}\n{documents[0].text[:360]} ..."
)

"""
### Initialize the Baidu VectorDB

Creation of the vector store entails creation of the underlying database collection if it does not exist yet:
"""
logger.info("### Initialize the Baidu VectorDB")

vector_store = BaiduVectorDB(
    endpoint="http://192.168.X.X",
    api_key="*******",
    table_params=TableParams(dimension=1536, drop_exists=True),
)

"""
Now wrap this store into an `index` LlamaIndex abstraction for later querying:
"""
logger.info("Now wrap this store into an `index` LlamaIndex abstraction for later querying:")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
Note that the above `from_documents` call does several things at once: it splits the input documents into chunks of manageable size ("nodes"), computes embedding vectors for each node, and stores them all in the Baidu VectorDB.

## Querying the store

### Basic querying
"""
logger.info("## Querying the store")

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")
logger.debug(response)

"""
### MMR-based queries

The MMR (maximal marginal relevance) method is designed to fetch text chunks from the store that are at the same time relevant to the query but as different as possible from each other, with the goal of providing a broader context to the building of the final answer:
"""
logger.info("### MMR-based queries")

query_engine = index.as_query_engine(vector_store_query_mode="mmr")
response = query_engine.query("Why did the author choose to work on AI?")
logger.debug(response)

"""
## Connecting to an existing store

Since this store is backed by Baidu VectorDB, it is persistent by definition. So, if you want to connect to a store that was created and populated previously, here is how:
"""
logger.info("## Connecting to an existing store")

vector_store = BaiduVectorDB(
    endpoint="http://192.168.X.X",
    api_key="*******",
    table_params=TableParams(dimension=1536, drop_exists=False),
)

new_index_instance = VectorStoreIndex.from_vector_store(
    vector_store=new_vector_store
)

query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(
    "What did the author study prior to working on AI?"
)
logger.debug(response)

"""
## Metadata filtering

The Baidu VectorDB vector store support metadata filtering in the form of exact-match `key=value` pairs at query time. The following cells, which work on a brand new collection, demonstrate this feature.

In this demo, for the sake of brevity, a single source document is loaded (the `./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/paul_graham_essay.txt` text file). Nevertheless, you will attach some custom metadata to the document to illustrate how you can can restrict queries with conditions on the metadata attached to the documents.
"""
logger.info("## Metadata filtering")

filter_fields = [
    TableField(name="source_type"),
]

md_storage_context = StorageContext.from_defaults(
    vector_store=BaiduVectorDB(
        endpoint="http://192.168.X.X",
        api_key="="*******",",
        table_params=TableParams(
            dimension=1536, drop_exists=True, filter_fields=filter_fields
        ),
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
    "./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data", file_metadata=my_file_metadata
).load_data()
md_index = VectorStoreIndex.from_documents(
    md_documents, storage_context=md_storage_context
)


md_query_engine = md_index.as_query_engine(
    filters=MetadataFilters(
        filters=[MetadataFilter(key="source_type", value="essay")]
    )
)
md_response = md_query_engine.query(
    "How long it took the author to write his thesis?"
)
logger.debug(md_response.response)

logger.info("\n\n[DONE]", bright=True)