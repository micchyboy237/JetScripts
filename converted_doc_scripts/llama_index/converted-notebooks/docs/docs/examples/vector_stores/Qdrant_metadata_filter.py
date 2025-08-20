from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import (
MetadataFilter,
MetadataFilters,
FilterOperator,
)
from llama_index.core.vector_stores import FilterOperator, FilterCondition
from llama_index.vector_stores.qdrant import QdrantVectorStore
import os
import qdrant_client
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/Qdrant_metadata_filter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Qdrant Vector Store - Metadata Filter

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Qdrant Vector Store - Metadata Filter")

# %pip install llama-index-vector-stores-qdrant

# !pip install llama-index qdrant_client

"""
Build the Qdrant VectorStore Client
"""
logger.info("Build the Qdrant VectorStore Client")


client = qdrant_client.QdrantClient(
    location=":memory:"
)

"""
Build the QdrantVectorStore and create a Qdrant Index
"""
logger.info("Build the QdrantVectorStore and create a Qdrant Index")


nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
            "year": 1994,
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
            "year": 1972,
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
            "theme": "Fiction",
            "year": 2010,
        },
    ),
    TextNode(
        text="To Kill a Mockingbird",
        metadata={
            "author": "Harper Lee",
            "theme": "Mafia",
            "year": 1960,
        },
    ),
    TextNode(
        text="1984",
        metadata={
            "author": "George Orwell",
            "theme": "Totalitarianism",
            "year": 1949,
        },
    ),
    TextNode(
        text="The Great Gatsby",
        metadata={
            "author": "F. Scott Fitzgerald",
            "theme": "The American Dream",
            "year": 1925,
        },
    ),
    TextNode(
        text="Harry Potter and the Sorcerer's Stone",
        metadata={
            "author": "J.K. Rowling",
            "theme": "Fiction",
            "year": 1997,
        },
    ),
]




# os.environ["OPENAI_API_KEY"] = "sk-..."


vector_store = QdrantVectorStore(
    client=client, collection_name="test_collection_1"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
Define metadata filters
"""
logger.info("Define metadata filters")


filters = MetadataFilters(
    filters=[
        MetadataFilter(key="theme", operator=FilterOperator.EQ, value="Mafia"),
    ]
)

"""
Retrieve from vector store with filters
"""
logger.info("Retrieve from vector store with filters")

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")

"""
Multiple Metadata Filters with `AND` condition
"""
logger.info("Multiple Metadata Filters with `AND` condition")


filters = MetadataFilters(
    filters=[
        MetadataFilter(key="theme", value="Fiction"),
        MetadataFilter(key="year", value=1997, operator=FilterOperator.GT),
    ],
    condition=FilterCondition.AND,
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("Harry Potter?")

"""
Use keyword arguments specific to Qdrant
"""
logger.info("Use keyword arguments specific to Qdrant")

retriever = index.as_retriever(
    vector_store_kwargs={"filter": {"theme": "Mafia"}}
)
retriever.retrieve("What is inception about?")

logger.info("\n\n[DONE]", bright=True)