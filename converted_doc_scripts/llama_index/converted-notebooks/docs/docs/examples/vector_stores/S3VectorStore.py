from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import (
VectorStoreIndex,
StorageContext,
SimpleDirectoryReader,
Document,
)
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
MetadataFilters,
MetadataFilter,
FilterOperator,
FilterCondition,
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.s3 import S3VectorStore
import boto3
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# S3VectorStore Integration

This is a vector store integration for LlamaIndex that uses S3Vectors.

[Find out more about S3Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html).

This notebook will assume that you have already created a S3 vector bucket (and possibly also an index).

## Installation
"""
logger.info("# S3VectorStore Integration")

# %pip install llama-index-vector-stores-s3 llama-index-embeddings-ollama

"""
## Usage

### Creating the vector store object

You can create a new vector index in an existing S3 bucket.

If you don't have S3 credentials configured in your environment, you can provide a boto3 session with credentials.
"""
logger.info("## Usage")


vector_store = S3VectorStore.create_index_from_bucket(
    bucket_name_or_arn="test-bucket",
    index_name="my-index",
    dimension=1536,
    distance_metric="cosine",
    data_type="float32",
    insert_batch_size=500,
    non_filterable_metadata_keys=["custom_field"],
)

"""
Or, you can use an existing vector index in an existing S3 bucket.
"""
logger.info("Or, you can use an existing vector index in an existing S3 bucket.")


vector_store = S3VectorStore(
    index_name_or_arn="my-index",
    bucket_name_or_arn="test-bucket",
    data_type="float32",
    distance_metric="cosine",
    insert_batch_size=500,
)

"""
### Using the vector store with an index

Once you have a vector store, you can use it with an index:
"""
logger.info("### Using the vector store with an index")



documents = [
    Document(text="Hello, world!", metadata={"key": "1"}),
    Document(text="Hello, world! 2", metadata={"key": "2"}),
]

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    embed_model=MLXEmbedding(model="mxbai-embed-large", api_key="..."),
)

nodes = index.as_retriever(similarity_top_k=1).retrieve("Hello, world!")
logger.debug(nodes[0].text)

"""
You can also use filters to help you retrieve the correct nodes!
"""
logger.info("You can also use filters to help you retrieve the correct nodes!")


nodes = index.as_retriever(
    similarity_top_k=2,
    filters=MetadataFilters(
        filters=[
            MetadataFilter(
                key="key",
                value="2",
                operator=FilterOperator.EQ,
            ),
        ],
        condition=FilterCondition.AND,
    ),
).retrieve("Hello, world!")

logger.debug(nodes[0].text)

"""
### Using the vector store directly

You can also use the vector store directly:
"""
logger.info("### Using the vector store directly")


nodes = [
    TextNode(text="Hello, world!"),
    TextNode(text="Hello, world! 2"),
]

embed_model = MLXEmbedding(model="mxbai-embed-large", api_key="...")
embeddings = embed_model.get_text_embedding_batch([n.text for n in nodes])
for node, embedding in zip(nodes, embeddings):
    node.embedding = embedding

vector_store.add(nodes)

query = VectorStoreQuery(
    query_embedding=embed_model.get_query_embedding("Hello, world!"),
    similarity_top_k=2,
)
results = vector_store.query(query)
logger.debug(results.nodes[0].text)

logger.info("\n\n[DONE]", bright=True)