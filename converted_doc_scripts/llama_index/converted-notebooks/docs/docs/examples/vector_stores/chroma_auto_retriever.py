from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import logging
import openai
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/chroma_auto_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Auto-Retrieval from a Vector Database

This guide shows how to perform **auto-retrieval** in LlamaIndex. 

Many popular vector dbs support a set of metadata filters in addition to a query string for semantic search. Given a natural language query, we first use the LLM to infer a set of metadata filters as well as the right query string to pass to the vector db (either can also be blank). This overall query bundle is then executed against the vector db.

This allows for more dynamic, expressive forms of retrieval beyond top-k semantic search. The relevant context for a given query may only require filtering on a metadata tag, or require a joint combination of filtering + semantic search within the filtered set, or just raw semantic search.

We demonstrate an example with Chroma, but auto-retrieval is also implemented with many other vector dbs (e.g. Pinecone, Weaviate, and more).

## Setup 

We first define imports and define an empty Chroma collection.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Auto-Retrieval from a Vector Database")

# %pip install llama-index-vector-stores-chroma

# !pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("MLX API Key:")

# openai.api_key = os.environ["OPENAI_API_KEY"]


chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

"""
## Defining Some Sample Data

We insert some sample nodes containing text chunks into the vector database. Note that each `TextNode` not only contains the text, but also metadata e.g. `category` and `country`. These metadata fields will get converted/stored as such in the underlying vector db.
"""
logger.info("## Defining Some Sample Data")



nodes = [
    TextNode(
        text=(
            "Michael Jordan is a retired professional basketball player,"
            " widely regarded as one of the greatest basketball players of all"
            " time."
        ),
        metadata={
            "category": "Sports",
            "country": "United States",
        },
    ),
    TextNode(
        text=(
            "Angelina Jolie is an American actress, filmmaker, and"
            " humanitarian. She has received numerous awards for her acting"
            " and is known for her philanthropic work."
        ),
        metadata={
            "category": "Entertainment",
            "country": "United States",
        },
    ),
    TextNode(
        text=(
            "Elon Musk is a business magnate, industrial designer, and"
            " engineer. He is the founder, CEO, and lead designer of SpaceX,"
            " Tesla, Inc., Neuralink, and The Boring Company."
        ),
        metadata={
            "category": "Business",
            "country": "United States",
        },
    ),
    TextNode(
        text=(
            "Rihanna is a Barbadian singer, actress, and businesswoman. She"
            " has achieved significant success in the music industry and is"
            " known for her versatile musical style."
        ),
        metadata={
            "category": "Music",
            "country": "Barbados",
        },
    ),
    TextNode(
        text=(
            "Cristiano Ronaldo is a Portuguese professional footballer who is"
            " considered one of the greatest football players of all time. He"
            " has won numerous awards and set multiple records during his"
            " career."
        ),
        metadata={
            "category": "Sports",
            "country": "Portugal",
        },
    ),
]

"""
## Build Vector Index with Chroma Vector Store

Here we load the data into the vector store. As mentioned above, both the text and metadata for each node will get converted into corresponding representations in Chroma. We can now run semantic queries and also metadata filtering on this data from Chroma.
"""
logger.info("## Build Vector Index with Chroma Vector Store")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
## Define `VectorIndexAutoRetriever`

We define our core `VectorIndexAutoRetriever` module. The module takes in `VectorStoreInfo`,
which contains a structured description of the vector store collection and the metadata filters it supports.
This information will then be used in the auto-retrieval prompt where the LLM infers metadata filters.
"""
logger.info("## Define `VectorIndexAutoRetriever`")



vector_store_info = VectorStoreInfo(
    content_info="brief biography of celebrities",
    metadata_info=[
        MetadataInfo(
            name="category",
            type="str",
            description=(
                "Category of the celebrity, one of [Sports, Entertainment,"
                " Business, Music]"
            ),
        ),
        MetadataInfo(
            name="country",
            type="str",
            description=(
                "Country of the celebrity, one of [United States, Barbados,"
                " Portugal]"
            ),
        ),
    ],
)
retriever = VectorIndexAutoRetriever(
    index, vector_store_info=vector_store_info
)

"""
## Running over some sample data

We try running over some sample data. Note how metadata filters are inferred - this helps with more precise retrieval!
"""
logger.info("## Running over some sample data")

retriever.retrieve("Tell me about two celebrities from United States")

retriever.retrieve("Tell me about Sports celebrities from United States")

logger.info("\n\n[DONE]", bright=True)