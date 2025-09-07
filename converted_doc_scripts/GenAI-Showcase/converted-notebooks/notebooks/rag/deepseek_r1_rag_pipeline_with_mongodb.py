from datasets import load_dataset
from jet.logger import CustomLogger
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd
import pymongo
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# DeepSeek and MongoDB For Movie Recommendation System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/deepseek_r1_rag_pipeline_with_mongodb.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)]()

## Install Libaries and Set Environment Variables
"""
logger.info("# DeepSeek and MongoDB For Movie Recommendation System")

# !pip install --quiet -U pymongo sentence-transformers datasets accelerate

# import getpass


def set_env_securely(var_name, prompt):
#     value = getpass.getpass(prompt)
    os.environ[var_name] = value

"""
## Step 1: Data Loading
"""
logger.info("## Step 1: Data Loading")


dataset = load_dataset("MongoDB/embedded_movies")

dataset_df = pd.DataFrame(dataset["train"])

dataset_df = dataset_df.dropna(subset=["fullplot"])
logger.debug("\nNumber of missing values in each column after removal:")
logger.debug(dataset_df.isnull().sum())

dataset_df = dataset_df.drop(columns=["plot_embedding"])

dataset_df.head()

"""
## Step 2: Generating Embeddings
"""
logger.info("## Step 2: Generating Embeddings")


embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def generate_embedding(text):
    return embedding_model.encode([text])[0].tolist()

dataset_df["embedding"] = dataset_df["fullplot"].apply(generate_embedding)

"""
## Step 3: MongoDB (Operational and Vector Database)

MongoDB acts as both an operational and a vector database for the RAG system.
MongoDB Atlas specifically provides a database solution that efficiently stores, queries and retrieves vector embeddings.

Creating a database and collection within MongoDB is made simple with MongoDB Atlas.

1. First, register for a [MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register). For existing users, sign into MongoDB Atlas.
2. [Follow the instructions](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/). Select Atlas UI as the procedure to deploy your first cluster.

Follow MongoDB’s [steps to get the connection](https://www.mongodb.com/docs/manual/reference/connection-string/) string from the Atlas UI. After setting up the database and obtaining the Atlas cluster connection URI, securely store the URI within your development environment.
"""
logger.info("## Step 3: MongoDB (Operational and Vector Database)")

set_env_securely("MONGO_URI", "Enter your MONGO URI: ")



def get_mongo_client(mongo_uri):
    """Establish and validate connection to the MongoDB."""

    client = pymongo.MongoClient(
        mongo_uri, appname="devrel.showcase.rag.deepseek_rag_movies.python"
    )

    ping_result = client.admin.command("ping")
    if ping_result.get("ok") == 1.0:
        logger.debug("Connection to MongoDB successful")
        return client
    else:
        logger.debug("Connection to MongoDB failed")
    return None


MONGO_URI = os.environ["MONGO_URI"]
if not MONGO_URI:
    logger.debug("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(MONGO_URI)

DB_NAME = "movies_database"
COLLECTION_NAME = "movies_collection"

db = mongo_client[DB_NAME]

collection = db[COLLECTION_NAME]

collection.delete_many({})

"""
## Step 4: Data Ingestion
"""
logger.info("## Step 4: Data Ingestion")

documents = dataset_df.to_dict("records")
collection.insert_many(documents)

logger.debug("Data ingestion into MongoDB completed")

"""
## Step 5: Vector Index Creation
"""
logger.info("## Step 5: Vector Index Creation")

embedding_field_name = "embedding"
vector_search_index_name = "vector_index"




def setup_vector_search_index(collection, index_definition, index_name="vector_index"):
    """
    Setup a vector search index for a MongoDB collection and wait for 30 seconds.

    Args:
    collection: MongoDB collection object
    index_definition: Dictionary containing the index definition
    index_name: Name of the index (default: "vector_index")
    """
    new_vector_search_index_model = SearchIndexModel(
        definition=index_definition, name=index_name, type="vectorSearch"
    )

    try:
        result = collection.create_search_index(model=new_vector_search_index_model)
        logger.debug(f"Creating index '{index_name}'...")

        logger.debug(f"Waiting for 30 seconds to allow index '{index_name}' to be created...")
        time.sleep(30)

        logger.debug(f"30-second wait completed for index '{index_name}'.")
        return result

    except Exception as e:
        logger.debug(f"Error creating new vector search index '{index_name}': {e!s}")
        return None

def create_vector_index_definition(dimensions):
    return {
        "fields": [
            {
                "type": "vector",
                "path": embedding_field_name,
                "numDimensions": dimensions,
                "similarity": "cosine",
            }
        ]
    }

DIMENSIONS = 384
vector_index_definition = create_vector_index_definition(dimensions=DIMENSIONS)

setup_vector_search_index(collection, vector_index_definition, "vector_index")

"""
## Step 6: Vector Search Function
"""
logger.info("## Step 6: Vector Search Function")

def vector_search(user_query, top_k=150):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    query_embedding = generate_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": top_k,  # Number of candidate matches to consider
            "limit": 5,  # Return top 4 matches
        }
    }

    project_stage = {
        "$project": {
            "_id": 0,  # Exclude the _id field
            "fullplot": 1,  # Include the plot field
            "title": 1,  # Include the title field
            "genres": 1,  # Include the genres field
            "score": {"$meta": "vectorSearchScore"},  # Include the search score
        }
    }

    pipeline = [vector_search_stage, project_stage]

    results = collection.aggregate(pipeline)
    return list(results)

"""
## Step 7: Semantic Search
"""
logger.info("## Step 7: Semantic Search")

query = "What are the some interesting action movies to watch that include business?"

get_knowledge = vector_search(query)

pd.DataFrame(get_knowledge).head()

logger.debug(f"\nTop 5 results for query '{query}':")

for result in get_knowledge:
    logger.debug(f"Title: {result['title']}, Score: {result['score']:.4f}")

"""
## Step 8: Retrieval Augmented Generation(RA)

Load DeepSeek model from Hugging Face
"""
logger.info("## Step 8: Retrieval Augmented Generation(RA)")


tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map="cuda"
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

model.to("cuda")

def rag_query(query):
    query = (
        "What are the some interesting action movies to watch that include business?"
    )

    get_knowledge = vector_search(query)

    combined_information = f"Query: {query}\nContinue to answer the query by using the Search Results:\n{get_knowledge}."

    input_ids = tokenizer(combined_information, return_tensors="pt").to("cuda")
    response = model.generate(**input_ids, max_new_tokens=1000)

    return tokenizer.decode(response[0], skip_special_tokens=False)

logger.debug(
    rag_query(
        "What's a romantic movie that I can watch with my wife? Make your response concise"
    )
)

logger.info("\n\n[DONE]", bright=True)