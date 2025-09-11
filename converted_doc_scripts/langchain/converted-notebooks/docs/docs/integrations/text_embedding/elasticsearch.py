from elasticsearch import Elasticsearch
from jet.logger import logger
from langchain_elasticsearch import ElasticsearchEmbeddings
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
Walkthrough of how to generate embeddings using a hosted embedding model in Elasticsearch

The easiest way to instantiate the `ElasticsearchEmbeddings` class it either
- using the `from_credentials` constructor if you are using Elastic Cloud
- or using the `from_es_connection` constructor with any Elasticsearch cluster
"""
logger.info("# Elasticsearch")

# !pip -q install langchain-elasticsearch


model_id = "your_model_id"

"""
## Testing with `from_credentials`
This required an Elastic Cloud `cloud_id`
"""
logger.info("## Testing with `from_credentials`")

embeddings = ElasticsearchEmbeddings.from_credentials(
    model_id,
    es_cloud_id="your_cloud_id",
    es_user="your_user",
    es_password="your_password",
)

documents = [
    "This is an example document.",
    "Another example document to generate embeddings for.",
]
document_embeddings = embeddings.embed_documents(documents)

for i, embedding in enumerate(document_embeddings):
    logger.debug(f"Embedding for document {i + 1}: {embedding}")

query = "This is a single query."
query_embedding = embeddings.embed_query(query)

logger.debug(f"Embedding for query: {query_embedding}")

"""
## Testing with Existing Elasticsearch client connection
This can be used with any Elasticsearch deployment
"""
logger.info("## Testing with Existing Elasticsearch client connection")


es_connection = Elasticsearch(
    hosts=["https://es_cluster_url:port"], basic_auth=("user", "password")
)

embeddings = ElasticsearchEmbeddings.from_es_connection(
    model_id,
    es_connection,
)

documents = [
    "This is an example document.",
    "Another example document to generate embeddings for.",
]
document_embeddings = embeddings.embed_documents(documents)

for i, embedding in enumerate(document_embeddings):
    logger.debug(f"Embedding for document {i + 1}: {embedding}")

query = "This is a single query."
query_embedding = embeddings.embed_query(query)

logger.debug(f"Embedding for query: {query_embedding}")

logger.info("\n\n[DONE]", bright=True)