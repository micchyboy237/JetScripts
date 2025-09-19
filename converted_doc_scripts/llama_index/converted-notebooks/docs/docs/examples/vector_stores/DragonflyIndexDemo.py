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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/DragonflyIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Dragonfly and Vector Store

In this notebook we are going to show a quick demo of using the Dragonfly with Vector Store.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Dragonfly and Vector Store")

# %pip install -U llama-index llama-index-vector-stores-redis llama-index-embeddings-cohere llama-index-embeddings-huggingface

# import getpass

warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


"""
### Start Dragonfly

The easiest way to start Dragonfly is using the Dragonfly docker image or
quickly signing up for a [Dragonfly Cloud](https://www.dragonflydb.io/cloud) demo instance.

To follow every step of this tutorial, launch the image as follows:

```bash
docker run -d -p 6379:6379 --name dragonfly docker.dragonflydb.io/dragonflydb/dragonfly
```

### Setup OllamaFunctionCalling
Lets first begin by adding the openai api key. This will allow us to access openai for embeddings and to use chatgpt.
"""
logger.info("### Start Dragonfly")

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
Here we will use a set of Paul Graham essays to provide the text to turn into embeddings, store in a vector store and query to find context for our LLM QnA loop.
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
### Initialize the default vector store

Now we have our documents prepared, we can initialize the vector store with **default** settings. This will allow us to store our vectors in Dragonfly and create an index for real-time search.
"""
logger.info("### Initialize the default vector store")


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

With Dragonfly, this is as simple as defining an index schema object
(from file or dict) and passing it through to the vector store client wrapper.

For this example, we will:
1. switch the embedding model to [Cohere](https://cohere.com/)
2. add an additional metadata field for the document `updated_at` timestamp
3. index the existing `file_name` metadata field
"""
logger.info("### Use a custom index schema")


# co_api_key = getpass.getpass("Cohere API Key:")

Settings.embed_model = CohereEmbedding(api_key=co_api_key)



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
Now that we have additional metadata indexed in Dragonfly, let's try some queries with filters.
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
However, the index still exists (with no associated documents).
"""
logger.info("However, the index still exists (with no associated documents).")

vector_store.index_exists()

vector_store.delete_index()

logger.debug("Number of documents after deleting", redis_client.dbsize())

logger.info("\n\n[DONE]", bright=True)