from jet.models.config import MODELS_CACHE_DIR
from datasets import load_dataset
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import (
FilterCondition,
FilterOperator,
MetadataFilter,
MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel
from typing import List
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# How To Build An AI Agent With Ollama, LlamaIndex and MongoDB

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/airbnb_agent_ollama_llamaindex_mongodb.ipynb)

## Install Libraries
"""
logger.info("# How To Build An AI Agent With Ollama, LlamaIndex and MongoDB")

# !pip install -qU llama-index  # main llamaindex libary
# !pip install -qU llama-index-vector-stores-mongodb # mongodb vector database
# !pip install -qU llama-index-llms-ollama # ollama llm provider
# !pip install -qU llama-index-embeddings-huggingface # ollama embedding provider
# !pip install -qU pymongo pandas datasets # others

"""
## Setup Prerequisites
"""
logger.info("## Setup Prerequisites")

# import getpass


# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Ollama API Key:")

# MONGODB_URI = getpass.getpass("Enter your MongoDB URI: ")
mongodb_client = MongoClient(
    MONGODB_URI, appname="devrel.content.airbnb_agent_mongodb_llamaindex"
)

"""
## Configure LLMs and Embedding Models
"""
logger.info("## Configure LLMs and Embedding Models")


Settings.embed_model = HuggingFaceEmbedding(
    model="mxbai-embed-large",
    dimensions=256,
    embed_batch_size=10,
#     ollama_api_key=os.environ["OPENAI_API_KEY"],
)
llm = Ollama(model="llama3.2", log_dir=f"{LOG_DIR}/chats", temperature=0)

"""
## Download the Dataset
"""
logger.info("## Download the Dataset")


data = load_dataset("MongoDB/airbnb_embeddings", split="train", streaming=True)
data = data.take(200)

data_df = pd.DataFrame(data)

data_df.head(5)

"""
## Data Processing
"""
logger.info("## Data Processing")


docs = data_df.to_dict(orient="records")

llama_documents = []
fields_to_include = [
    "amenities",
    "address",
    "availability",
    "review_scores",
    "listing_url",
]

for doc in docs:
    metadata = {key: doc[key] for key in fields_to_include}
    llama_doc = Document(text=doc["description"], metadata=metadata)
    llama_documents.append(llama_doc)

llama_documents[0]

"""
## Create MongoDB Atlas Vector Store
"""
logger.info("## Create MongoDB Atlas Vector Store")


DB_NAME = "airbnb"
COLLECTION_NAME = "listings_reviews"
VS_INDEX_NAME = "vector_index"
FTS_INDEX_NAME = "fts_index"
collection = mongodb_client[DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    mongodb_client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    vector_index_name=VS_INDEX_NAME,
    fulltext_index_name=FTS_INDEX_NAME,
    embedding_key="embedding",
    text_key="text",
)
vector_store_context = StorageContext.from_defaults(vector_store=vector_store)
vector_store_index = VectorStoreIndex.from_documents(
    llama_documents, storage_context=vector_store_context, show_progress=True
)

"""
## Create Vector and Full-text Search Indexes
"""
logger.info("## Create Vector and Full-text Search Indexes")


vs_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 256,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "metadata.amenities"},
            {"type": "filter", "path": "metadata.review_scores.review_scores_rating"},
        ]
    },
    name=VS_INDEX_NAME,
    type="vectorSearch",
)

fts_model = SearchIndexModel(
    definition={"mappings": {"dynamic": False, "fields": {"text": {"type": "string"}}}},
    name=FTS_INDEX_NAME,
    type="search",
)

for model in [vs_model, fts_model]:
    try:
        collection.create_search_index(model=model)
        logger.debug(f"Successfully created index for model {model}.")
    except OperationFailure:
        logger.debug(f"Duplicate index found for model {model}. Skipping index creation.")

"""
## Creating Retriever Tool for the Agent
"""
logger.info("## Creating Retriever Tool for the Agent")



def get_airbnb_listings(query: str, amenities: List[str]) -> str:
    """
    Provides information about Airbnb listings.

    query (str): User query
    amenities (List[str]): List of amenities
    rating (int): Listing rating
    """
    filters = [
        MetadataFilter(
            key="metadata.review_scores.review_scores_rating",
            value=80,
            operator=FilterOperator.GTE,
        )
    ]
    amenities_filter = [
        MetadataFilter(
            key="metadata.amenities", value=amenity, operator=FilterOperator.EQ
        )
        for amenity in amenities
    ]
    filters.extend(amenities_filter)

    filters = MetadataFilters(
        filters=filters,
        condition=FilterCondition.AND,
    )

    query_engine = vector_store_index.as_query_engine(
        similarity_top_k=5, vector_store_query_mode="hybrid", alpha=0.7, filters=filters
    )
    response = query_engine.query(query)
    nodes = response.source_nodes
    listings = [node.metadata["listing_url"] for node in nodes]
    return listings

query_tool = FunctionTool.from_defaults(
    name="get_airbnb_listings", fn=get_airbnb_listings
)

"""
## Create the AI Agent
"""
logger.info("## Create the AI Agent")


agent_worker = FunctionCallingAgentWorker.from_tools(
    [query_tool], llm=llm, verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query("Give me listings in Porto with a Waterfront.")

logger.info("\n\n[DONE]", bright=True)