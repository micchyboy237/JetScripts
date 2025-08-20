from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate.embedded import EmbeddedOptions
import logging
import openai
import os
import shutil
import sys
import weaviate


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/WeaviateIndex_auto_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Auto-Retrieval from a Weaviate Vector Database

This guide shows how to perform **auto-retrieval** in LlamaIndex with [Weaviate](https://weaviate.io/). 

The Weaviate vector database supports a set of [metadata filters](https://weaviate.io/developers/weaviate/search/filters) in addition to a query string for semantic search. Given a natural language query, we first use a Large Language Model (LLM) to infer a set of metadata filters as well as the right query string to pass to the vector database (either can also be blank). This overall query bundle is then executed against the vector database.

This allows for more dynamic, expressive forms of retrieval beyond top-k semantic search. The relevant context for a given query may only require filtering on a metadata tag, or require a joint combination of filtering + semantic search within the filtered set, or just raw semantic search.

## Setup 

We first define imports and define an empty Weaviate collection.

If you're opening this Notebook on Colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Auto-Retrieval from a Weaviate Vector Database")

# %pip install llama-index-vector-stores-weaviate

# !pip install llama-index weaviate-client


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
We will be using GPT-4 for its reasoning capabilities to infer the metadata filters. Depending on your use case, `"gpt-3.5-turbo"` can work as well.
"""
logger.info("We will be using GPT-4 for its reasoning capabilities to infer the metadata filters. Depending on your use case, `"gpt-3.5-turbo"` can work as well.")

# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("MLX API Key:")
# openai.api_key = os.environ["OPENAI_API_KEY"]


Settings.llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
Settings.embed_model = MLXEmbedding()

"""
This Notebook uses Weaviate in [Embedded mode](https://weaviate.io/developers/weaviate/installation/embedded), which is supported on Linux and macOS.

If you prefer to try out Weaviate's fully managed service, [Weaviate Cloud Services (WCS)](https://weaviate.io/developers/weaviate/installation/weaviate-cloud-services), you can enable the code in the comments.
"""
logger.info("This Notebook uses Weaviate in [Embedded mode](https://weaviate.io/developers/weaviate/installation/embedded), which is supported on Linux and macOS.")


client = weaviate.connect_to_embedded()

"""

cluster_url = ""
api_key = ""

client = weaviate.connect_to_wcs(cluster_url=cluster_url,
    auth_credentials=weaviate.auth.AuthApiKey(api_key),
)

"""

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
## Build Vector Index with Weaviate Vector Store

Here we load the data into the vector store. As mentioned above, both the text and metadata for each node will get converted into corresponding representations in Weaviate. We can now run semantic queries and also metadata filtering on this data from Weaviate.
"""
logger.info("## Build Vector Index with Weaviate Vector Store")


vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="LlamaIndex_filter"
)

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

response = retriever.retrieve("Tell me about celebrities from United States")

logger.debug(response[0])

response = retriever.retrieve(
    "Tell me about Sports celebrities from United States"
)

logger.debug(response[0])

logger.info("\n\n[DONE]", bright=True)