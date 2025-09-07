from datasets import load_dataset
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import (
FilterCondition,
FilterOperator,
MetadataFilter,
MetadataFilters,
)
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
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
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/advanced_techniques/retrieval_strategies_mongodb_llamaindex_togetherai.ipynb)

# Optimizing for relevance using MongoDB, LlamaIndex and Together.ai

In this notebook, we will explore and tune different retrieval options in MongoDB's LlamaIndex integration using Together.ai to get the most relevant results.


"""
logger.info("# Optimizing for relevance using MongoDB, LlamaIndex and Together.ai")

# !curl ipinfo.io

"""
## Step 1: Install libraries

- **pymongo**: Python package to interact with MongoDB databases and collections
<p>
- **llama-index**: Python package for the LlamaIndex LLM framework
<p>
- **llama-index-llms-together**: Python package to use TogetherAI models via their LlamaIndex integration
<p>
- **llama-index-vector-stores-mongodb**: Python package for MongoDB’s LlamaIndex integration
"""
logger.info("## Step 1: Install libraries")

# !pip install -qU pymongo llama-index llama-index-vector-stores-mongodb together \
llama-index-llms-together llama-index-embeddings-together datasets

"""
## Step 2: Setup prerequisites

- **Set the MongoDB connection string**: Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.

- **Set the Together.ai API key**: Steps to obtain an API key as [here](https://docs.together.ai/reference/authentication-1)
"""
logger.info("## Step 2: Setup prerequisites")

# import getpass


# os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter your Together.AI API key: ")

# MONGODB_URI = getpass.getpass("Enter your MongoDB URI: ")
mongodb_client = MongoClient(
    MONGODB_URI, appname="devrel.showcase.retrieval_strategies_llamaindex"
)

"""
## Step 3: Load and process the dataset
"""
logger.info("## Step 3: Load and process the dataset")


data = load_dataset("MongoDB/embedded_movies", split="train")
data = pd.DataFrame(data)

data.head()

data = data.fillna({"genres": "[]", "languages": "[]", "cast": "[]", "imdb": "{}"})

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
## Step 4: Define the LLM and Embedding Model
"""
logger.info("## Step 4: Define the LLM and Embedding Model")

Settings.llm = llm = TogetherLLM(model="mistralai/Mistral-7B-v0.1")

Settings.embed_model = embed_model = TogetherEmbedding(
    model_name="togethercomputer/m2-bert-80M-2k-retrieval"
)

"""
## Step 5: Create MongoDB Atlas Vector store
"""
logger.info("## Step 5: Create MongoDB Atlas Vector store")

VS_INDEX_NAME = "vector_index"
FTS_INDEX_NAME = "fts_index"
DB_NAME = "mdb_llamaindex_together"
COLLECTION_NAME = "hybrid_search"
collection = mongodb_client[DB_NAME][COLLECTION_NAME]

"""
### Create Atlas Vector Index
"""
logger.info("### Create Atlas Vector Index")

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
    vector_store_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_store_index = VectorStoreIndex.from_documents(
        documents, storage_context=vector_store_context, show_progress=True
    )

"""
### Create Atlas Search Index
"""
logger.info("### Create Atlas Search Index")

vs_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 768,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "metadata.rating"},
            {"type": "filter", "path": "metadata.languages"},
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
    except OperationFailure:
        logger.debug(f"Duplicate index found for model {model}. Skipping index creation.")

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
### Query
"""
logger.info("### Query")

logger.debug("\nTop 5 highest rated movies (rating >= 8.0):")

top_movies = list(
    collection.find(
        {"metadata.rating": {"$gte": 8.0}},  # filter
        {"metadata.title": 1, "metadata.rating": 1, "_id": 0},  # projection
    )
    .sort("metadata.rating", -1)  # sort
    .limit(5)
)  # limit

for movie in top_movies:
    logger.debug(f"Title: {movie['metadata']['title']}, Rating: {movie['metadata']['rating']}")

"""
### Aggregate
"""
logger.info("### Aggregate")

pipeline = [
    {"$unwind": "$metadata.languages"},
    {
        "$group": {
            "_id": "$metadata.languages",
            "average_rating": {"$avg": "$metadata.rating"},
        }
    },
    {"$sort": {"average_rating": -1}},
    {"$limit": 10},
]

results = collection.aggregate(pipeline)
logger.debug("\nAverage ratings by language:")
for result in results:
    logger.debug(f"Language: {result['_id']}, Average Rating: {result['average_rating']:.2f}")

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

get_recommendations(query="Action movies about humans fighting machines", mode="hybrid")

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
        MetadataFilter(key="metadata.rating", value=7, operator=FilterOperator.GT),
        MetadataFilter(
            key="metadata.languages", value="English", operator=FilterOperator.EQ
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