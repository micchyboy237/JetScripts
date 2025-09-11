from jet.logger import logger
from langchain_core.globals import set_llm_cache
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.cache import MongoDBAtlasSemanticCache
from langchain_mongodb.cache import MongoDBCache
from langchain_mongodb.retrievers import MongoDBAtlasFullTextSearchRetriever
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
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
# MongoDB Atlas

>[MongoDB Atlas](https://www.mongodb.com/docs/atlas/) is a fully-managed cloud
> database available in AWS, Azure, and GCP.  It now has support for native
> Vector Search on the MongoDB document data.

## Installation and Setup

See [detail configuration instructions](/docs/integrations/vectorstores/mongodb_atlas).

We need to install `langchain-mongodb` python package.
"""
logger.info("# MongoDB Atlas")

pip install langchain-mongodb

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/mongodb_atlas).
"""
logger.info("## Vector Store")


"""
## Retrievers

### Full Text Search Retriever

>`Hybrid Search Retriever` performs full-text searches using
> Lucene’s standard (`BM25`) analyzer.
"""
logger.info("## Retrievers")


"""
### Hybrid Search Retriever

>`Hybrid Search Retriever` combines vector and full-text searches weighting
> them the via `Reciprocal Rank Fusion` (`RRF`) algorithm.
"""
logger.info("### Hybrid Search Retriever")


"""
## Model Caches

### MongoDBCache

An abstraction to store a simple cache in MongoDB. This does not use Semantic Caching, nor does it require an index to be made on the collection before generation.

To import this cache:
"""
logger.info("## Model Caches")


"""
To use this cache with your LLMs:
"""
logger.info("To use this cache with your LLMs:")



mongodb_atlas_uri = "<YOUR_CONNECTION_STRING>"
COLLECTION_NAME="<YOUR_CACHE_COLLECTION_NAME>"
DATABASE_NAME="<YOUR_DATABASE_NAME>"

set_llm_cache(MongoDBCache(
    connection_string=mongodb_atlas_uri,
    collection_name=COLLECTION_NAME,
    database_name=DATABASE_NAME,
))

"""
### MongoDBAtlasSemanticCache
Semantic caching allows users to retrieve cached prompts based on semantic similarity between the user input and previously cached results. Under the hood it blends MongoDBAtlas as both a cache and a vectorstore.
The MongoDBAtlasSemanticCache inherits from `MongoDBAtlasVectorSearch` and needs an Atlas Vector Search Index defined to work. Please look at the [usage example](/docs/integrations/vectorstores/mongodb_atlas) on how to set up the index.

To import this cache:
"""
logger.info("### MongoDBAtlasSemanticCache")


"""
To use this cache with your LLMs:
"""
logger.info("To use this cache with your LLMs:")



mongodb_atlas_uri = "<YOUR_CONNECTION_STRING>"
COLLECTION_NAME="<YOUR_CACHE_COLLECTION_NAME>"
DATABASE_NAME="<YOUR_DATABASE_NAME>"

set_llm_cache(MongoDBAtlasSemanticCache(
    embedding=FakeEmbeddings(),
    connection_string=mongodb_atlas_uri,
    collection_name=COLLECTION_NAME,
    database_name=DATABASE_NAME,
))

logger.info("\n\n[DONE]", bright=True)