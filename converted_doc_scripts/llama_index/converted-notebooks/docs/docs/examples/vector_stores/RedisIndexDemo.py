from datetime import datetime
from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import (
MetadataFilters,
MetadataFilter,
ExactMatchFilter,
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from redis import Redis
from redisvl.schema import IndexSchema
import logging
import os
import shutil
import sys
import textwrap
import warnings


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/RedisIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Redis Vector Store

In this notebook we are going to show a quick demo of using the RedisVectorStore.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Redis Vector Store")

# %pip install -U llama-index llama-index-vector-stores-redis llama-index-embeddings-cohere llama-index-embeddings-huggingface

# import getpass

warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


"""
### Start Redis

The easiest way to start Redis is using the [Redis Stack](https://hub.docker.com/r/redis/redis-stack) docker image or
quickly signing up for a [FREE Redis Cloud](https://redis.com/try-free) instance.

To follow every step of this tutorial, launch the image as follows:

```bash
docker run --name redis-vecdb -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

This will also launch the RedisInsight UI on port 8001 which you can view at http://localhost:8001.

### Setup OllamaFunctionCalling
Lets first begin by adding the openai api key. This will allow us to access openai for embeddings and to use chatgpt.
"""
logger.info("### Start Redis")

# oai_api_key = getpass.getpass("OllamaFunctionCalling API Key:")
# os.environ["OPENAI_API_KEY"] = oai_api_key

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Read in a dataset
Here we will use a set of Paul Graham essays to provide the text to turn into embeddings, store in a ``RedisVectorStore`` and query to find context for our LLM QnA loop.
"""
logger.info("### Read in a dataset")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
logger.debug(
    "Document ID:",
    documents[0].id_,
    "Document Filename:",
    documents[0].metadata["file_name"],
)

"""
### Initialize the default Redis Vector Store

Now we have our documents prepared, we can initialize the Redis Vector Store with **default** settings. This will allow us to store our vectors in Redis and create an index for real-time search.
"""
logger.info("### Initialize the default Redis Vector Store")


redis_client = Redis.from_url("redis://localhost:6379")

vector_store = RedisVectorStore(redis_client=redis_client, overwrite=True)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Query the default vector store

Now that we have our data stored in the index, we can ask questions against the index.

The index will use the data as the knowledge base for an LLM. The default setting for as_query_engine() utilizes OllamaFunctionCalling embeddings and GPT as the language model. Therefore, an OllamaFunctionCalling key is required unless you opt for a customized or local language model.

Below we will test searches against out index and then full RAG with an LLM.
"""
logger.info("### Query the default vector store")

query_engine = index.as_query_engine()
retriever = index.as_retriever()

result_nodes = retriever.retrieve("What did the author learn?")
for node in result_nodes:
    logger.debug(node)

response = query_engine.query("What did the author learn?")
logger.debug(textwrap.fill(str(response), 100))

result_nodes = retriever.retrieve("What was a hard moment for the author?")
for node in result_nodes:
    logger.debug(node)

response = query_engine.query("What was a hard moment for the author?")
logger.debug(textwrap.fill(str(response), 100))

index.vector_store.delete_index()

"""
### Use a custom index schema

In most use cases, you need the ability to customize the underling index configuration
and specification. For example, this is handy in order to define specific metadata filters you wish to enable.

With Redis, this is as simple as defining an index schema object
(from file or dict) and passing it through to the vector store client wrapper.

For this example, we will:
1. switch the embedding model to [Cohere](cohereai.com)
2. add an additional metadata field for the document `updated_at` timestamp
3. index the existing `file_name` metadata field
"""
logger.info("### Use a custom index schema")


# co_api_key = getpass.getpass("Cohere API Key:")
os.environ["CO_API_KEY"] = co_api_key

Settings.embed_model = CohereEmbedding()



custom_schema = IndexSchema.from_dict(
    {
        "index": {
            "name": "paul_graham",
            "prefix": "essay",
            "key_separator": ":",
        },
        "fields": [
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            {"type": "numeric", "name": "updated_at"},
            {"type": "tag", "name": "file_name"},
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 1024,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    }
)

custom_schema.index

custom_schema.fields

"""
Learn more about [schema and index design](https://redisvl.com) with redis.
"""
logger.info("Learn more about [schema and index design](https://redisvl.com) with redis.")



def date_to_timestamp(date_string: str) -> int:
    date_format: str = "%Y-%m-%d"
    return int(datetime.strptime(date_string, date_format).timestamp())


for document in documents:
    document.metadata["updated_at"] = date_to_timestamp(
        document.metadata["last_modified_date"]
    )

vector_store = RedisVectorStore(
    schema=custom_schema,  # provide customized schema
    redis_client=redis_client,
    overwrite=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Query the vector store and filter on metadata
Now that we have additional metadata indexed in Redis, let's try some queries with filters.
"""
logger.info("### Query the vector store and filter on metadata")


retriever = index.as_retriever(
    similarity_top_k=3,
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(key="file_name", value="paul_graham_essay.txt"),
            MetadataFilter(
                key="updated_at",
                value=date_to_timestamp("2023-01-01"),
                operator=">=",
            ),
            MetadataFilter(
                key="text",
                value="learn",
                operator="text_match",
            ),
        ],
        condition="and",
    ),
)

result_nodes = retriever.retrieve("What did the author learn?")

for node in result_nodes:
    logger.debug(node)

"""
### Restoring from an existing index in Redis
Restoring from an index requires a Redis connection client (or URL), `overwrite=False`, and passing in the same schema object used before. (This can be offloaded to a YAML file for convenience using `.to_yaml()`)
"""
logger.info("### Restoring from an existing index in Redis")

custom_schema.to_yaml("paul_graham.yaml")

vector_store = RedisVectorStore(
    schema=IndexSchema.from_yaml("paul_graham.yaml"),
    redis_client=redis_client,
)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

"""
**In the near future** -- we will implement a convenience method to load just using an index name:
```python
RedisVectorStore.from_existing_index(index_name="paul_graham", redis_client=redis_client)
```

### Deleting documents or index completely

Sometimes it may be useful to delete documents or the entire index. This can be done using the `delete` and `delete_index` methods.
"""
logger.info("### Deleting documents or index completely")

document_id = documents[0].doc_id
document_id

logger.debug("Number of documents before deleting", redis_client.dbsize())
vector_store.delete(document_id)
logger.debug("Number of documents after deleting", redis_client.dbsize())

"""
However, the Redis index still exists (with no associated documents) for continuous upsert.
"""
logger.info("However, the Redis index still exists (with no associated documents) for continuous upsert.")

vector_store.index_exists()

vector_store.delete_index()

logger.debug("Number of documents after deleting", redis_client.dbsize())

"""
### Troubleshooting

If you get an empty query result, there a couple of issues to check:

#### Schema

Unlike other vector stores, Redis expects users to explicitly define the schema for the index. This is for a few reasons:
1. Redis is used for many use cases, including real-time vector search, but also for standard document storage/retrieval, caching, messaging, pub/sub, session mangement, and more. Not all attributes on records need to be indexed for search. This is partially an efficiency thing, and partially an attempt to minimize user foot guns.
2. All index schemas, when using Redis & LlamaIndex, must include the following fields `id`, `doc_id`, `text`, and `vector`, at a minimum.

Instantiate your `RedisVectorStore` with the default schema (assumes OllamaFunctionCalling embeddings), or with a custom schema (see above).

#### Prefix issues

Redis expects all records to have a key prefix that segments the keyspace into "partitions"
for potentially different applications, use cases, and clients.

Make sure that the chosen `prefix`, as part of the index schema, is consistent across your code (tied to a specific index).

To see what prefix your index was created with, you can run `FT.INFO <name of your index>` in the Redis CLI and look under `index_definition` => `prefixes`.

#### Data vs Index
Redis treats the records in the dataset and the index as different entities. This allows you more flexibility in performing updates, upserts, and index schema migrations.

If you have an existing index and want to make sure it's dropped, you can run `FT.DROPINDEX <name of your index>` in the Redis CLI. Note that this will *not* drop your actual data unless you pass `DD`

#### Empty queries when using metadata

If you add metadata to the index *after* it has already been created and then try to query over that metadata, your queries will come back empty.

Redis indexes fields upon index creation only (similar to how it indexes the prefixes, above).
"""
logger.info("### Troubleshooting")

logger.info("\n\n[DONE]", bright=True)