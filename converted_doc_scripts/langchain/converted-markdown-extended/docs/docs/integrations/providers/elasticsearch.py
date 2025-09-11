from jet.logger import logger
from langchain.chains.elasticsearch_database import ElasticsearchDatabaseChain
from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain_community.vectorstores.ecloud_vector_search import EcloudESVectorStore
from langchain_elasticsearch import ElasticsearchCache
from langchain_elasticsearch import ElasticsearchChatMessageHistory
from langchain_elasticsearch import ElasticsearchEmbeddings
from langchain_elasticsearch import ElasticsearchEmbeddingsCache
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_elasticsearch import ElasticsearchStore
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
# Elasticsearch

> [Elasticsearch](https://www.elastic.co/elasticsearch/) is a distributed, RESTful search and analytics engine.
> It provides a distributed, multi-tenant-capable full-text search engine with an HTTP web interface and schema-free
> JSON documents.

## Installation and Setup

### Setup Elasticsearch

There are two ways to get started with Elasticsearch:

#### Install Elasticsearch on your local machine via Docker

Example: Run a single-node Elasticsearch instance with security disabled.
This is not recommended for production use.
"""
logger.info("# Elasticsearch")

docker run -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.security.http.ssl.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.9.0

"""
#### Deploy Elasticsearch on Elastic Cloud

`Elastic Cloud` is a managed Elasticsearch service. Signup for a [free trial](https://cloud.elastic.co/registration?utm_source=langchain&utm_content=documentation).

### Install Client
"""
logger.info("#### Deploy Elasticsearch on Elastic Cloud")

pip install elasticsearch
pip install langchain-elasticsearch

"""
## Embedding models

See a [usage example](/docs/integrations/text_embedding/elasticsearch).
"""
logger.info("## Embedding models")


"""
## Vector store

See a [usage example](/docs/integrations/vectorstores/elasticsearch).
"""
logger.info("## Vector store")


"""
### Third-party integrations

#### EcloudESVectorStore
"""
logger.info("### Third-party integrations")


"""
## Retrievers

### ElasticsearchRetriever

The `ElasticsearchRetriever` enables flexible access to all Elasticsearch features
through the Query DSL.

See a [usage example](/docs/integrations/retrievers/elasticsearch_retriever).
"""
logger.info("## Retrievers")


"""
### BM25

See a [usage example](/docs/integrations/retrievers/elastic_search_bm25).
"""
logger.info("### BM25")


"""
## Memory

See a [usage example](/docs/integrations/memory/elasticsearch_chat_message_history).
"""
logger.info("## Memory")


"""
## LLM cache

See a [usage example](/docs/integrations/llm_caching/#elasticsearch-caches).
"""
logger.info("## LLM cache")


"""
## Byte Store

See a [usage example](/docs/integrations/stores/elasticsearch).
"""
logger.info("## Byte Store")


"""
## Chain

It is a chain for interacting with Elasticsearch Database.
"""
logger.info("## Chain")


logger.info("\n\n[DONE]", bright=True)