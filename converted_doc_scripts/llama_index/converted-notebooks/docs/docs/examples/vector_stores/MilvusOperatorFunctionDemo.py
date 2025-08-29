from jet.logger import CustomLogger
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import (
MetadataFilter,
MetadataFilters,
FilterOperator,
)
from llama_index.core.vector_stores import FilterCondition
from llama_index.vector_stores.milvus import MilvusVectorStore
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/MilvusOperatorFunctionDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Milvus Vector Store - Metadata Filter

This notebook illustrates the use of the Milvus vector store in LlamaIndex, focusing on metadata filtering capabilities. You will learn how to index documents with metadata, perform vector searches with LlamaIndex's built-in metadata filters, and apply Milvus's native filtering expressions to the vector store.

By the end of this notebook, you will understand how to utilize Milvus's filtering features to narrow down search results based on document metadata.

## Prerequisites

**Install dependencies**

Before getting started, make sure you have the following dependencies installed:
"""
logger.info("# Milvus Vector Store - Metadata Filter")

# ! pip install llama-index-vector-stores-milvus llama-index

"""
> If you're using Google Colab, you may need to **restart the runtime** (Navigate to the "Runtime" menu at the top of the interface, and select "Restart session" from the dropdown menu.)

**Set up accounts**

This tutorial uses OllamaFunctionCallingAdapter for text embeddings and answer generation. You need to prepare the [OllamaFunctionCallingAdapter API key](https://platform.openai.com/api-keys).
"""
logger.info("This tutorial uses OllamaFunctionCallingAdapter for text embeddings and answer generation. You need to prepare the [OllamaFunctionCallingAdapter API key](https://platform.openai.com/api-keys).")


openai.api_key = "sk-"

"""
To use the Milvus vector store, specify your Milvus server `URI` (and optionally with the `TOKEN`). To start a Milvus server, you can set up a Milvus server by following the [Milvus installation guide](https://milvus.io/docs/install-overview.md) or simply trying [Zilliz Cloud](https://docs.zilliz.com/docs/register-with-zilliz-cloud) for free.
"""
logger.info("To use the Milvus vector store, specify your Milvus server `URI` (and optionally with the `TOKEN`). To start a Milvus server, you can set up a Milvus server by following the [Milvus installation guide](https://milvus.io/docs/install-overview.md) or simply trying [Zilliz Cloud](https://docs.zilliz.com/docs/register-with-zilliz-cloud) for free.")

URI = "./milvus_filter_demo.db"  # Use Milvus-Lite for demo purpose

"""
**Prepare data**

For this example, we'll use a few books with similar or identical titles but different metadata (author, genre, and publication year) as the sample data. This will help demonstrate how Milvus can filter and retrieve documents based on both vector similarity and metadata attributes.
"""
logger.info("For this example, we'll use a few books with similar or identical titles but different metadata (author, genre, and publication year) as the sample data. This will help demonstrate how Milvus can filter and retrieve documents based on both vector similarity and metadata attributes.")


nodes = [
    TextNode(
        text="Life: A User's Manual",
        metadata={
            "author": "Georges Perec",
            "genre": "Postmodern Fiction",
            "year": 1978,
        },
    ),
    TextNode(
        text="Life and Fate",
        metadata={
            "author": "Vasily Grossman",
            "genre": "Historical Fiction",
            "year": 1980,
        },
    ),
    TextNode(
        text="Life",
        metadata={
            "author": "Keith Richards",
            "genre": "Memoir",
            "year": 2010,
        },
    ),
    TextNode(
        text="The Life",
        metadata={
            "author": "Malcolm Knox",
            "genre": "Literary Fiction",
            "year": 2011,
        },
    ),
]

"""
## Build Index

In this section, we will store sample data in Milvus using the default embedding model (OllamaFunctionCallingAdapter's `text-embedding-ada-002`). Titles will be converted into text embeddings and stored in a dense embedding field, while all metadata will be stored in scalar fields.
"""
logger.info("## Build Index")



vector_store = MilvusVectorStore(
    uri=URI,
    collection_name="test_filter_collection",  # Change collection name here
    dim=1536,  # Vector dimension depends on the embedding model
    overwrite=True,  # Drop collection if exists
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
## Metadata Filters

In this section, we will apply LlamaIndex's built-in metadata filters and conditions to Milvus search.

**Define metadata filters**
"""
logger.info("## Metadata Filters")


filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="year", value=2000, operator=FilterOperator.GT
        )  # year > 2000
    ]
)

"""
**Retrieve from vector store with filters**
"""

retriever = index.as_retriever(filters=filters, similarity_top_k=5)
result_nodes = retriever.retrieve("Books about life")
for node in result_nodes:
    logger.debug(node.text)
    logger.debug(node.metadata)
    logger.debug("\n")

"""
### Multiple Metdata Filters

You can also combine multiple metadata filters to create more complex queries. LlamaIndex supports both `AND` and `OR` conditions to combine filters. This allows for more precise and flexible retrieval of documents based on their metadata attributes.

**Condition `AND`**

Try an example filtering for books published between 1979 and 2010 (specifically, where 1979 < year ≤ 2010):
"""
logger.info("### Multiple Metdata Filters")


filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="year", value=1979, operator=FilterOperator.GT
        ),  # year > 1979
        MetadataFilter(
            key="year", value=2010, operator=FilterOperator.LTE
        ),  # year <= 2010
    ],
    condition=FilterCondition.AND,
)

retriever = index.as_retriever(filters=filters, similarity_top_k=5)
result_nodes = retriever.retrieve("Books about life")
for node in result_nodes:
    logger.debug(node.text)
    logger.debug(node.metadata)
    logger.debug("\n")

"""
**Condition `OR`**

Try another example that filters books written by either Georges Perec or Keith Richards:
"""
logger.info("Try another example that filters books written by either Georges Perec or Keith Richards:")

filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="author", value="Georges Perec", operator=FilterOperator.EQ
        ),  # author is Georges Perec
        MetadataFilter(
            key="author", value="Keith Richards", operator=FilterOperator.EQ
        ),  # author is Keith Richards
    ],
    condition=FilterCondition.OR,
)

retriever = index.as_retriever(filters=filters, similarity_top_k=5)
result_nodes = retriever.retrieve("Books about life")
for node in result_nodes:
    logger.debug(node.text)
    logger.debug(node.metadata)
    logger.debug("\n")

"""
## Use Milvus's Keyword Arguments

In addition to the built-in filtering capabilities, you can use Milvus's native filtering expressions by the `string_expr` keyword argument. This allows you to pass specific filter expressions directly to Milvus during search operations, extending beyond the standard metadata filtering to access Milvus's advanced filtering capabilities.

Milvus provides powerful and flexible filtering options that enable precise querying of your vector data:

- Basic Operators: Comparison operators, range filters, arithmetic operators, and logical operators
- Filter Expression Templates: Predefined patterns for common filtering scenarios
- Specialized Operators: Data type-specific operators for JSON or array fields

For comprehensive documentation and examples of Milvus filtering expressions, refer to the official documentation of [Milvus Filtering](https://milvus.io/docs/boolean.md).
"""
logger.info("## Use Milvus's Keyword Arguments")

retriever = index.as_retriever(
    vector_store_kwargs={
        "string_expr": "genre like '%Fiction'",
    },
    similarity_top_k=5,
)
result_nodes = retriever.retrieve("Books about life")
for node in result_nodes:
    logger.debug(node.text)
    logger.debug(node.metadata)
    logger.debug("\n")

logger.info("\n\n[DONE]", bright=True)