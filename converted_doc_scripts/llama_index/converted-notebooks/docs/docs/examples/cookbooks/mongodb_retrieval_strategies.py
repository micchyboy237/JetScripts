from jet.models.config import MODELS_CACHE_DIR
from datasets import load_dataset
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import (
MetadataFilter,
MetadataFilters,
FilterOperator,
FilterCondition,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/mongodb_retrieval_strategies.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/optimize-relevance-mongodb-llamaindex/?utm_campaign=devrel&utm_source=cross-post&utm_medium=organic_social&utm_content=https%3A%2F%2Fgithub.com%2Fmongodb-developer%2FGenAI-Showcase&utm_term=apoorva.joshi)

# Optimizing for relevance using MongoDB and LlamaIndex

In this notebook, we will explore and tune different retrieval options in MongoDB's LlamaIndex integration to get the most relevant results.

## Step 1: Install libraries

- **pymongo**: Python package to interact with MongoDB databases and collections
<p>
- **llama-index**: Python package for the LlamaIndex LLM framework
<p>
- **llama-index-llms-ollama**: Python package to use OllamaFunctionCalling models via their LlamaIndex integration 
<p>
- **llama-index-vector-stores-mongodb**: Python package for MongoDB’s LlamaIndex integration
"""
logger.info("# Optimizing for relevance using MongoDB and LlamaIndex")

# !pip install -qU pymongo llama-index llama-index-llms-ollama llama-index-vector-stores-mongodb

"""
## Step 2: Setup prerequisites

- **Set the MongoDB connection string**: Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.

- **Set the OllamaFunctionCalling API key**: Steps to obtain an API key as [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
"""
logger.info("## Step 2: Setup prerequisites")

# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OllamaFunctionCalling API key: ")

# MONGODB_URI = getpass.getpass("Enter your MongoDB URI: ")
mongodb_client = MongoClient(
    MONGODB_URI, appname="devrel.content.retrieval_strategies_llamaindex"
)

"""
## Step 3: Load and process the dataset
"""
logger.info("## Step 3: Load and process the dataset")


data = load_dataset("MongoDB/embedded_movies", split="train")
data = pd.DataFrame(data)

data.head()

data = data.fillna(
    {"genres": "[]", "languages": "[]", "cast": "[]", "imdb": "{}"}
)

documents = []

for _, row in data.iterrows():
    title = row["title"]
    rating = row["imdb"].get("rating", 0)
    languages = row["languages"]
    cast = row["cast"]
    genres = row["genres"]
    metadata = {"title": title, "rating": rating, "languages": languages}
    text = f"Title: {title}\nPlot: {row['fullplot']}\nCast: {', '.join(item for item in cast)}\nGenres: {', '.join(item for item in  genres)}\nLanguages: {', '.join(item for item in languages)}\nRating: {rating}"
    documents.append(Document(text=text, metadata=metadata))

logger.debug(documents[0].text)

logger.debug(documents[0].metadata)

"""
## Step 4: Create MongoDB Atlas vector store
"""
logger.info("## Step 4: Create MongoDB Atlas vector store")


Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

VS_INDEX_NAME = "vector_index"
FTS_INDEX_NAME = "fts_index"
DB_NAME = "llamaindex"
COLLECTION_NAME = "hybrid_search"
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
if collection.count_documents({}) > 0:
    vector_store_index = VectorStoreIndex.from_vector_store(vector_store)
else:
    vector_store_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    vector_store_index = VectorStoreIndex.from_documents(
        documents, storage_context=vector_store_context, show_progress=True
    )

"""
## Step 5: Create Atlas Search indexes
"""
logger.info("## Step 5: Create Atlas Search indexes")

vs_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1536,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "metadata.rating"},
            {"type": "filter", "path": "metadata.language"},
        ]
    },
    name=VS_INDEX_NAME,
    type="vectorSearch",
)

fts_model = SearchIndexModel(
    definition={
        "mappings": {"dynamic": False, "fields": {"text": {"type": "string"}}}
    },
    name=FTS_INDEX_NAME,
    type="search",
)

for model in [vs_model, fts_model]:
    try:
        collection.create_search_index(model=model)
    except OperationFailure:
        logger.debug(
            f"Duplicate index found for model {model}. Skipping index creation."
        )

"""
## Step 6: Get movie recommendations
"""
logger.info("## Step 6: Get movie recommendations")

def get_recommendations(query: str, mode: str, **kwargs) -> None:
    """
    Get movie recommendations

    Args:
        query (str): User query
        mode (str): Retrieval mode. One of (default, text_search, hybrid)
    """
    query_engine = vector_store_index.as_query_engine(
        similarity_top_k=5, vector_store_query_mode=mode, **kwargs
    )
    response = query_engine.query(query)
    nodes = response.source_nodes
    for node in nodes:
        title = node.metadata["title"]
        rating = node.metadata["rating"]
        score = node.score
        logger.debug(f"Title: {title} | Rating: {rating} | Relevance Score: {score}")

"""
### Full-text search
"""
logger.info("### Full-text search")

get_recommendations(
    query="Action movies about humans fighting machines",
    mode="text_search",
)

"""
### Vector search
"""
logger.info("### Vector search")

get_recommendations(
    query="Action movies about humans fighting machines", mode="default"
)

"""
### Hybrid search
"""
logger.info("### Hybrid search")

get_recommendations(
    query="Action movies about humans fighting machines", mode="hybrid"
)

get_recommendations(
    query="Action movies about humans fighting machines",
    mode="hybrid",
    alpha=0.7,
)

get_recommendations(
    query="Action movies about humans fighting machines",
    mode="hybrid",
    alpha=0.3,
)

"""
### Combining metadata filters with search
"""
logger.info("### Combining metadata filters with search")


filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="metadata.rating", value=7, operator=FilterOperator.GT
        ),
        MetadataFilter(
            key="metadata.languages",
            value="English",
            operator=FilterOperator.EQ,
        ),
    ],
    condition=FilterCondition.AND,
)

get_recommendations(
    query="Action movies about humans fighting machines",
    mode="hybrid",
    alpha=0.7,
    filters=filters,
)

logger.info("\n\n[DONE]", bright=True)