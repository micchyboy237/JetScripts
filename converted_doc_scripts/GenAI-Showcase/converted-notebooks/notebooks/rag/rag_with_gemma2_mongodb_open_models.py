from datasets import load_dataset
from jet.logger import CustomLogger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd
import pymongo
import shutil
import torch


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# RAG Pipeline With Gemma 2, MongoDB and Hugging Face [Open Models]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/rag_with_gemma2_mongodb_open_models.ipynb)

## Set Up Libraries
"""
logger.info("# RAG Pipeline With Gemma 2, MongoDB and Hugging Face [Open Models]")

# !pip install --upgrade --quiet datasets pandas pymongo sentence_transformers
# !pip install --upgrade --quiet transformers
# !pip install --upgrade --quiet accelerate

"""
## Data Loading
"""
logger.info("## Data Loading")



dataset = load_dataset("MongoDB/embedded_movies", split="train", streaming=True)
dataset = dataset.take(4000)

dataset_df = pd.DataFrame(dataset)

dataset_df.head(5)

"""
## Data Cleaning
"""
logger.info("## Data Cleaning")

dataset_df = dataset_df.dropna(subset=["fullplot"])
logger.debug("\nNumber of missing values in each column after removal:")
logger.debug(dataset_df.isnull().sum())

dataset_df = dataset_df.drop(columns=["plot_embedding"])
dataset_df.head(5)

"""
## Embedding Generation
"""
logger.info("## Embedding Generation")


embedding_model = SentenceTransformer("thenlper/gte-large")


def get_embedding(text: str) -> list[float]:
    if not text.strip():
        logger.debug("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)

    return embedding.tolist()


tqdm.pandas(desc="Generating embeddings")
dataset_df["embedding"] = dataset_df["fullplot"].progress_apply(get_embedding)

dataset_df.head()

"""
## MongoDB Vector Database and Connection Setup

MongoDB acts as both an operational and a vector database for the RAG system.
MongoDB Atlas specifically provides a database solution that efficiently stores, queries and retrieves vector embeddings.

Creating a database and collection within MongoDB is made simple with MongoDB Atlas.

1. First, register for a [MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register). For existing users, sign into MongoDB Atlas.
2. [Follow the instructions](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/). Select Atlas UI as the procedure to deploy your first cluster.
3. Create the database: `movie_rec_sys`.
4. Within the database ` movie_collection`, create the collection ‘listings_reviews’.
5. Create a [vector search index](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure/) named vector_index for the ‘listings_reviews’ collection. This index enables the RAG application to retrieve records as additional context to supplement user queries via vector search. Below is the JSON definition of the data collection vector search index.

Your vector search index created on MongoDB Atlas should look like below:

```
{
  "fields": [
    {
      "numDimensions": 1024,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}

```

Follow MongoDB’s [steps to get the connection](https://www.mongodb.com/docs/manual/reference/connection-string/) string from the Atlas UI. After setting up the database and obtaining the Atlas cluster connection URI, securely store the URI within your development environment.
"""
logger.info("## MongoDB Vector Database and Connection Setup")


os.environ["MONGO_URI"] = ""



def get_mongo_client(mongo_uri):
    """Establish and validate connection to the MongoDB."""

    client = pymongo.MongoClient(mongo_uri, appname="devrel.showcase.gemma2.python")

    ping_result = client.admin.command("ping")
    if ping_result.get("ok") == 1.0:
        logger.debug("Connection to MongoDB successful")
        return client
    logger.debug("Connection to MongoDB failed")
    return None


mongo_uri = os.environ["MONGO_URI"]

if not mongo_uri:
    logger.debug("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

DB_NAME = "movie_rec_sys"
COLLECTION_NAME = "movie_collection"

db = mongo_client.get_database(DB_NAME)
collection = db.get_collection(COLLECTION_NAME)

collection.delete_many({})

"""
## Data Ingestion
"""
logger.info("## Data Ingestion")

documents = dataset_df.to_dict("records")
collection.insert_many(documents)

logger.debug("Data ingestion into MongoDB completed")

"""
## Vector Search Operation
"""
logger.info("## Vector Search Operation")

def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": 4,  # Return top 4 matches
        }
    }

    unset_stage = {
        "$unset": "embedding"  # Exclude the 'embedding' field from the results
    }

    project_stage = {
        "$project": {
            "_id": 0,  # Exclude the _id field
            "fullplot": 1,  # Include the plot field
            "title": 1,  # Include the title field
            "genres": 1,  # Include the genres field
            "score": {
                "$meta": "vectorSearchScore"  # Include the search score
            },
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]

    results = collection.aggregate(pipeline)
    return list(results)

"""
## Handle User Results
"""
logger.info("## Handle User Results")

def get_search_result(query, collection):
    get_knowledge = vector_search(query, collection)

    search_result = ""
    for result in get_knowledge:
        search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('fullplot', 'N/A')}\n"

    return search_result

query = "What is the best romantic movie to watch and why?"
source_information = get_search_result(query, collection)
combined_information = f"Query: {query}\nContinue to answer the query by using the Search Results:\n{source_information}."

logger.debug(combined_information)

"""
## Load Gemma 2
"""
logger.info("## Load Gemma 2")


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it", device_map="auto", torch_dtype=torch.bfloat16
)

input_ids = tokenizer(combined_information, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_length=500)
logger.debug(tokenizer.decode(outputs[0]))

logger.info("\n\n[DONE]", bright=True)