import os
from jet.llm.ollama.base import Ollama
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.indices.managed.vectara import VectaraAutoRetriever
from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.core.indices.managed.types import ManagedIndexQueryMode
from llama_index.core.schema import TextNode
import sys
import logging
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/vectara_auto_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Auto-Retrieval from a Vectara Index
#
# This guide shows how to perform **auto-retrieval** in LlamaIndex with Vectara.
#
# With Auto-retrieval we interpret a retrieval query before submitting it to Vectara to identify potential rewrites of the query as a shorter query along with some metadata filtering.
#
# For example, a query like "what is the revenue in 2022" might be rewritten as "what is the revenue" along with a filter of "doc.year = 2022". Let's see how this works via an example.

# Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# !pip install llama_index llama-index-llms-ollama llama-index-indices-managed-vectara


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Defining Some Sample Data
#
# We first define a dataset of movies:
# 1. Each node describes a movie.
# 2. The `text` describes the movie, whereas `metadata` defines certain metadata fields like year, director, rating or genre.
#
# In Vectara you will need to [define](https://docs.vectara.com/docs/learn/metadata-search-filtering/filter-overview) these metadata fields in your coprus as filterable attributes so that filtering can occur with them.

nodes = [
    TextNode(
        text=(
            "A pragmatic paleontologist touring an almost complete theme park on an island "
            + "in Central America is tasked with protecting a couple of kids after a power "
            + "failure causes the park's cloned dinosaurs to run loose."
        ),
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    TextNode(
        text=(
            "A thief who steals corporate secrets through the use of dream-sharing technology "
            + "is given the inverse task of planting an idea into the mind of a C.E.O., "
            + "but his tragic past may doom the project and his team to disaster."
        ),
        metadata={
            "year": 2010,
            "director": "Christopher Nolan",
            "rating": 8.2,
        },
    ),
    TextNode(
        text="Barbie suffers a crisis that leads her to question her world and her existence.",
        metadata={
            "year": 2023,
            "director": "Greta Gerwig",
            "genre": "fantasy",
            "rating": 9.5,
        },
    ),
    TextNode(
        text=(
            "A cowboy doll is profoundly threatened and jealous when a new spaceman action "
            + "figure supplants him as top toy in a boy's bedroom."
        ),
        metadata={"year": 1995, "genre": "animated", "rating": 8.3},
    ),
    TextNode(
        text=(
            "When Woody is stolen by a toy collector, Buzz and his friends set out on a "
            + "rescue mission to save Woody before he becomes a museum toy property with his "
            + "roundup gang Jessie, Prospector, and Bullseye. "
        ),
        metadata={"year": 1999, "genre": "animated", "rating": 7.9},
    ),
    TextNode(
        text=(
            "The toys are mistakenly delivered to a day-care center instead of the attic "
            + "right before Andy leaves for college, and it's up to Woody to convince the "
            + "other toys that they weren't abandoned and to return home."
        ),
        metadata={"year": 2010, "genre": "animated", "rating": 8.3},
    ),
]

# Then we load our sample data into our Vectara Index.


os.environ["VECTARA_API_KEY"] = "<YOUR_VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_ID"] = "<YOUR_VECTARA_CORPUS_ID>"
os.environ["VECTARA_CUSTOMER_ID"] = "<YOUR_VECTARA_CUSTOMER_ID>"

index = VectaraIndex(nodes=nodes)

# Defining the `VectorStoreInfo`
#
# We define a `VectorStoreInfo` object, which contains a structured description of the metadata filters suported by our Vectara Index. This information is later on usedin the auto-retrieval prompt, enabling the LLM to infer the metadata filters to use for a specific query.

vector_store_info = VectorStoreInfo(
    content_info="information about a movie",
    metadata_info=[
        MetadataInfo(
            name="genre",
            description="""
                The genre of the movie. 
                One of ['science fiction', 'fantasy', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']
            """,
            type="string",
        ),
        MetadataInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        MetadataInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        MetadataInfo(
            name="rating",
            description="A 1-10 rating for the movie",
            type="float",
        ),
    ],
)

# Running auto-retrieval
# Now let's create a `VectaraAutoRetriever` instance and try `retrieve()`:


llm = Ollama(model="llama3.2", request_timeout=300.0,
             context_window=4096, temperature=0)

retriever = VectaraAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    llm=llm,
    verbose=True,
)

retriever.retrieve("movie directed by Greta Gerwig")

retriever.retrieve("a movie with a rating above 8")

# We can also include standard `VectaraRetriever` arguments in the `VectaraAutoRetriever`. For example, if we want to include a `filter` that would be added to any additional filtering from the query itself, we can do it as follows:

logger.info("\n\n[DONE]", bright=True)
