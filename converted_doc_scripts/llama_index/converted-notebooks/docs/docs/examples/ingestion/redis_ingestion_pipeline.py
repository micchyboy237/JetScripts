from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import (
DocstoreStrategy,
IngestionPipeline,
IngestionCache,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Redis Ingestion Pipeline

This walkthrough shows how to use Redis for both the vector store, cache, and docstore in an Ingestion Pipeline.

## Dependencies

Install and start redis, setup MLX API key
"""
logger.info("# Redis Ingestion Pipeline")

# %pip install llama-index-storage-docstore-redis
# %pip install llama-index-vector-stores-redis
# %pip install llama-index-embeddings-huggingface

# !docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest


# os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
## Create Seed Data
"""
logger.info("## Create Seed Data")

# !rm -rf test_redis_data
# !mkdir -p test_redis_data
# !echo "This is a test file: one!" > test_redis_data/test1.txt
# !echo "This is a test file: two!" > test_redis_data/test2.txt


documents = SimpleDirectoryReader(
    "./test_redis_data", filename_as_id=True
).load_data()

"""
## Run the Redis-Based Ingestion Pipeline

With a vector store attached, the pipeline will handle upserting data into your vector store.

However, if you only want to handle duplcates, you can change the strategy to `DUPLICATES_ONLY`.
"""
logger.info("## Run the Redis-Based Ingestion Pipeline")




embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

custom_schema = IndexSchema.from_dict(
    {
        "index": {"name": "redis_vector_store", "prefix": "doc"},
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

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    docstore=RedisDocumentStore.from_host_and_port(
        "localhost", 6379, namespace="document_store"
    ),
    vector_store=RedisVectorStore(
        schema=custom_schema,
        redis_url="redis://localhost:6379",
    ),
    cache=IngestionCache(
        cache=RedisCache.from_host_and_port("localhost", 6379),
        collection="redis_cache",
    ),
    docstore_strategy=DocstoreStrategy.UPSERTS,
)

nodes = pipeline.run(documents=documents)
logger.debug(f"Ingested {len(nodes)} Nodes")

"""
## Confirm documents are ingested

We can create a vector index using our vector store, and quickly ask which documents are seen.
"""
logger.info("## Confirm documents are ingested")


index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, embed_model=embed_model
)

logger.debug(
    index.as_query_engine(similarity_top_k=10).query(
        "What documents do you see?"
    )
)

"""
## Add data and Ingest

Here, we can update an existing file, as well as add a new one!
"""
logger.info("## Add data and Ingest")

# !echo "This is a test file: three!" > test_redis_data/test3.txt
# !echo "This is a NEW test file: one!" > test_redis_data/test1.txt

documents = SimpleDirectoryReader(
    "./test_redis_data", filename_as_id=True
).load_data()

nodes = pipeline.run(documents=documents)

logger.debug(f"Ingested {len(nodes)} Nodes")

index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, embed_model=embed_model
)

response = index.as_query_engine(similarity_top_k=10).query(
    "What documents do you see?"
)

logger.debug(response)

for node in response.source_nodes:
    logger.debug(node.get_text())

"""
As we can see, the data was deduplicated and upserted correctly! Only three nodes are in the index, even though we ran the full pipeline twice.
"""
logger.info("As we can see, the data was deduplicated and upserted correctly! Only three nodes are in the index, even though we ran the full pipeline twice.")

logger.info("\n\n[DONE]", bright=True)