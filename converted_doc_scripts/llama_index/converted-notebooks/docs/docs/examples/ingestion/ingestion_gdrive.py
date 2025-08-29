from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import (
DocstoreStrategy,
IngestionPipeline,
IngestionCache,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.google import GoogleDriveReader
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Building a Live RAG Pipeline over Google Drive Files

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/ingestion/ingestion_gdrive.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In this guide we show you how to build a "live" RAG pipeline over Google Drive files.

This pipeline will index Google Drive files and dump them to a Redis vector store. Afterwards, every time you rerun the ingestion pipeline, the pipeline will propagate **incremental updates**, so that only changed documents are updated in the vector store. This means that we don't re-index all the documents!

We use the following [data source](https://drive.google.com/drive/folders/1RFhr3-KmOZCR5rtp4dlOMNl3LKe1kOA5?usp=sharing) - you will need to copy these files and upload them to your own Google Drive directory! 

**NOTE**: You will also need to setup a service account and credentials.json. See our LlamaHub page for the Google Drive loader for more details: https://llamahub.ai/l/readers/llama-index-readers-google?from=readers

## Setup

We install required packages and launch the Redis Docker image.
"""
logger.info("# Building a Live RAG Pipeline over Google Drive Files")

# %pip install llama-index-storage-docstore-redis
# %pip install llama-index-vector-stores-redis
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-readers-google

# !docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
## Define Ingestion Pipeline

Here we define the ingestion pipeline. Given a set of documents, we will run sentence splitting/embedding transformations, and then load them into a Redis docstore/vector store.

The vector store is for indexing the data + storing the embeddings, the docstore is for tracking duplicates.
"""
logger.info("## Define Ingestion Pipeline")



embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

custom_schema = IndexSchema.from_dict(
    {
        "index": {"name": "gdrive", "prefix": "doc"},
        "fields": [
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    }
)

vector_store = RedisVectorStore(
    schema=custom_schema,
    redis_url="redis://localhost:6379",
)

if vector_store.index_exists():
    vector_store.delete_index()

cache = IngestionCache(
    cache=RedisCache.from_host_and_port("localhost", 6379),
    collection="redis_cache",
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    docstore=RedisDocumentStore.from_host_and_port(
        "localhost", 6379, namespace="document_store"
    ),
    vector_store=vector_store,
    cache=cache,
    docstore_strategy=DocstoreStrategy.UPSERTS,
)

"""
### Define our Vector Store Index

We define our index to wrap the underlying vector store.
"""
logger.info("### Define our Vector Store Index")


index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, embed_model=embed_model
)

"""
## Load Initial Data

Here we load data from our [Google Drive Loader](https://llamahub.ai/l/readers/llama-index-readers-google?from=readers) on LlamaHub. 

The loaded docs are the header sections of our [Use Cases from our documentation](https://docs.llamaindex.ai/en/latest/use_cases/q_and_a/root.html).
"""
logger.info("## Load Initial Data")


loader = GoogleDriveReader()

def load_data(folder_id: str):
    docs = loader.load_data(folder_id=folder_id)
    for doc in docs:
        doc.id_ = doc.metadata["file_name"]
    return docs


docs = load_data(folder_id="1RFhr3-KmOZCR5rtp4dlOMNl3LKe1kOA5")

nodes = pipeline.run(documents=docs)
logger.debug(f"Ingested {len(nodes)} Nodes")

"""
Since this is our first time starting up the vector store, we see that we've transformed/ingested all the documents into it (by chunking, and then by embedding).

### Ask Questions over Initial Data
"""
logger.info("### Ask Questions over Initial Data")

query_engine = index.as_query_engine()

response = query_engine.query("What are the sub-types of question answering?")

logger.debug(str(response))

"""
## Modify and Reload the Data

Let's try modifying our ingested data! 

We modify the "Q&A" doc to include an extra "structured analytics" block of text. See our [updated document](https://docs.google.com/document/d/1QQMKNAgyplv2IUOKNClEBymOFaASwmsZFoLmO_IeSTw/edit?usp=sharing) as a reference.

Now let's rerun the ingestion pipeline.
"""
logger.info("## Modify and Reload the Data")

docs = load_data(folder_id="1RFhr3-KmOZCR5rtp4dlOMNl3LKe1kOA5")
nodes = pipeline.run(documents=docs)
logger.debug(f"Ingested {len(nodes)} Nodes")

"""
Notice how only one node is ingested. This is beacuse only one document changed, while the other documents stayed the same. This means that we only need to re-transform and re-embed one document!

### Ask Questions over New Data
"""
logger.info("### Ask Questions over New Data")

query_engine = index.as_query_engine()

response = query_engine.query("What are the sub-types of question answering?")

logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)