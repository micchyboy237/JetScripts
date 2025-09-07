from datasets import load_dataset
from google.colab import userdata
from jet.logger import CustomLogger
import ollama
import os
import pandas as pd
import pymongo
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/ollama_text_3_emebdding.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/using-ollama-latest-embeddings-rag-system-mongodb/)

# Using Ollama Latest Embeddings In A RAG System With MongoDB

Ollama recently released new embeddings and moderation models. This article explores the step-by-step implementation process of utilizing one of the new embedding models: mxbai-embed-large within a Retrieval Augmented Generation(RAG) System powered by MongoDB Atlas Vector Database.

## Step 1: Libraries Installation


Below are brief explanations of the tools and libraries utilised within the implementation code:
* **datasets**: This library is part of the Hugging Face ecosystem. By installing 'datasets', we gain access to a number of pre-processed and ready-to-use datasets, which are essential for training and fine-tuning machine learning models or benchmarking their performance.

* **pandas**: A data science library that provides robust data structures and methods for data manipulation, processing and analysis.

* **ollama**: This is the official Python client library for accessing Ollama's suite of AI models and tools, including GPT and embedding models.italicised text

* **pymongo**: PyMongo is a Python toolkit for MongoDB. It enables interactions with a MongoDB database.
"""
logger.info("# Using Ollama Latest Embeddings In A RAG System With MongoDB")

# !pip install datasets pandas ollama pymongo

"""
## Step 2: Data Loading


Load the dataset titled ["AIatMongoDB/embedded_movies"](https://huggingface.co/datasets/AIatMongoDB/embedded_movies). This dataset is a collection of movie-related details that include attributes such as the title, release year, cast, plot and more. A unique feature of this dataset is the plot_embedding field for each movie. These embeddings are generated using Ollama's text-embedding-ada-002 model.
"""
logger.info("## Step 2: Data Loading")


dataset = load_dataset("MongoDB/embedded_movies")

dataset_df = pd.DataFrame(dataset["train"])

dataset_df.head(5)

"""
## Step 3: Data Cleaning and Preparation

The next step cleans the data and prepares it for the next stage, which creates a new embedding data point using the new Ollama embedding model.
"""
logger.info("## Step 3: Data Cleaning and Preparation")

logger.debug("Columns:", dataset_df.columns)
logger.debug("\nNumber of rows and columns:", dataset_df.shape)
logger.debug("\nBasic Statistics for numerical data:")
logger.debug(dataset_df.describe())
logger.debug("\nNumber of missing values in each column:")
logger.debug(dataset_df.isnull().sum())

dataset_df = dataset_df.dropna(subset=["plot"])
logger.debug("\nNumber of missing values in each column after removal:")
logger.debug(dataset_df.isnull().sum())

dataset_df = dataset_df.drop(columns=["plot_embedding"])
dataset_df.head(5)

"""


## Step 4: Create embeddings with Ollama

This stage focuses on generating new embeddings using Ollama's advanced model.
This demonstration utilises a Google Colab Notebook, where environment variables are configured explicitly within the notebook's Secret section and accessed using the user data module. In a production environment, the environment variables that store secret keys are usually stored in a '.env' file or equivalent.

An [Ollama API](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-api-key) key is required to ensure the successful completion of this step. More details on Ollama's embedding models can be found on the official [site](https://platform.ollama.com/docs/guides/embeddings).
"""
logger.info("## Step 4: Create embeddings with Ollama")


ollama.api_key = userdata.get("ollama")

EMBEDDING_MODEL = "mxbai-embed-large"


def get_embedding(text):
    """Generate an embedding for the given text using Ollama's API."""

    if not text or not isinstance(text, str):
        return None

    try:
        embedding = (
            ollama.embeddings.create(input=text, model=EMBEDDING_MODEL)
            .data[0]
            .embedding
        )
        return embedding
    except Exception as e:
        logger.debug(f"Error in get_embedding: {e}")
        return None


dataset_df["plot_embedding_optimised"] = dataset_df["plot"].apply(get_embedding)

dataset_df.head()

"""
## Step 5: Vector Database Setup and Data Ingestion

MongoDB acts as both an operational and a vector database. It offers a database solution that efficiently stores, queries and retrieves vector embeddings—the advantages of this lie in the simplicity of database maintenance, management and cost.

**To create a new MongoDB database, set up a database cluster:**

1. Head over to MongoDB official site and register for a [free MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register), or for existing users, [sign into MongoDB Atlas](https://account.mongodb.com/account/login?nds=true).

2. Select the 'Database' option on the left-hand pane, which will navigate to the Database Deployment page, where there is a deployment specification of any existing cluster. Create a new database cluster by clicking on the "+Create" button.

3.   Select all the applicable configurations for the database cluster. Once all the configuration options are selected, click the “Create Cluster” button to deploy the newly created cluster. MongoDB also enables the creation of free clusters on the “Shared Tab”.

 *Note: Don’t forget to whitelist the IP for the Python host or 0.0.0.0/0 for any IP when creating proof of concepts.*

4. After successfully creating and deploying the cluster, the cluster becomes accessible on the ‘Database Deployment’ page.

5. Click on the “Connect” button of the cluster to view the option to set up a connection to the cluster via various language drivers.

6. This tutorial only requires the cluster's URI(unique resource identifier). Grab the URI and copy it into the Google Colabs Secrets environment in a variable named `MONGO_URI` or place it in a .env file or equivalent.
"""
logger.info("## Step 5: Vector Database Setup and Data Ingestion")



def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(
            mongo_uri, appname="devrel.showcase.rag_ollama_text_embedding_3"
        )
        logger.debug("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        logger.debug(f"Connection failed: {e}")
        return None


mongo_uri = userdata.get("MONGO_URI_2")
if not mongo_uri:
    logger.debug("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

db = mongo_client["movies"]
collection = db["movie_collection"]

collection.delete_many({})

documents = dataset_df.to_dict("records")
collection.insert_many(documents)

logger.debug("Data ingestion into MongoDB completed")

"""
## Step 6: Create a Vector Search Index

At this point make sure that your vector index is created via MongoDB Atlas.
Follow instructions here:

This next step is mandatory for conducting efficient and accurate vector-based searches based on the vector embeddings stored within the documents in the ‘movie_collection’ collection. Creating a Vector Search Index enables the ability to traverse the documents efficiently to retrieve documents with embeddings that match the query embedding based on vector similarity. Go here to read more about [MongoDB Vector Search Index](https://www.mongodb.com/docs/atlas/atlas-search/field-types/knn-vector/).

## Step 7: Perform Vector Search on User Queries

This step combines all the activities in the previous step to provide the functionality of conducting vector search on stored records based on embedded user queries.

This step implements a function that returns a vector search result by generating a query embedding and defining a MongoDB aggregation pipeline. The pipeline, consisting of the `$vectorSearch` and `$project` stages, queries using the generated vector and formats the results to include only required information like plot, title, and genres while incorporating a search score for each result.

This selective projection enhances query performance by reducing data transfer and optimizes the use of network and memory resources, which is especially critical when handling large datasets. For AI Engineers and Developers considering data security at an early stage, the chances of sensitive data leaked to the client side can be minimized by carefully excluding fields irrelevant to the user's query.
"""
logger.info("## Step 6: Create a Vector Search Index")

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

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "plot_embedding_optimised",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 5,  # Return top 5 matches
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude the _id field
                "plot": 1,  # Include the plot field
                "title": 1,  # Include the title field
                "genres": 1,  # Include the genres field
                "score": {"$meta": "vectorSearchScore"},  # Include the search score
            }
        },
    ]

    results = collection.aggregate(pipeline)
    return list(results)

"""
## Step 8: Handling User Query and Result

The final step in the implementation phase focuses on the practical application of our vector search functionality and AI integration to handle user queries effectively.

The handle_user_query function performs a vector search on the MongoDB collection based on the user's query and utilizes Ollama's GPT-3.5 model to generate context-aware responses.
"""
logger.info("## Step 8: Handling User Query and Result")

def handle_user_query(query, collection):
    get_knowledge = vector_search(query, collection)

    search_result = ""
    for result in get_knowledge:
        search_result += (
            f"Title: {result.get('title', 'N/A')}, Plot: {result.get('plot', 'N/A')}\n"
        )

    completion = ollama.chat.completions.create(
        model="llama3.2", log_dir=f"{LOG_DIR}/chats",
        messages=[
            {"role": "system", "content": "You are a movie recommendation system."},
            {
                "role": "user",
                "content": "Answer this user query: "
                + query
                + " with the following context: "
                + search_result,
            },
        ],
    )

    return (completion.choices[0].message.content), search_result

query = "What is the best romantic movie to watch?"
response, source_information = handle_user_query(query, collection)

logger.debug(f"Response: {response}")
logger.debug(f"Source Information: \n{source_information}")

logger.info("\n\n[DONE]", bright=True)