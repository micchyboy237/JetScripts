from datasets import load_dataset
from jet.logger import CustomLogger
import anthropic
import os
import pandas as pd
import pymongo
import shutil
import voyageai


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/anthropic_mongodb_pam_ai_stack.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/rag_with_claude_opus_mongodb/)
"""

# !pip install --quiet pymongo datasets pandas anthropic voyageai

"""
# Set Environment Variables
"""
logger.info("# Set Environment Variables")


# os.environ["ANTHROPIC_API_KEY"] = ""
# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

os.environ["VOYAGE_API_KEY"] = ""
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")

os.environ["HF_TOKEN"] = ""



dataset = load_dataset("MongoDB/tech-news-embeddings", split="train", streaming=True)
combined_df = dataset.take(500)

combined_df = pd.DataFrame(combined_df)

combined_df = combined_df.drop(columns=["_id"])

combined_df = combined_df.drop(columns=["embedding"])

combined_df.head()


vo = voyageai.Client(api_key=VOYAGE_API_KEY)


def get_embedding(text: str) -> list[float]:
    if not text.strip():
        logger.debug("Attempted to get embedding for empty text.")
        return []

    embedding = vo.embed(text, model="voyage-large-2", input_type="document")

    return embedding.embeddings[0]


combined_df["embedding"] = combined_df["description"].apply(get_embedding)

combined_df.head()

"""
Create Database and Collection
Create Vector Search Index
"""
logger.info("Create Database and Collection")

os.environ["MONGO_URI"] = ""



def get_mongo_client(mongo_uri):
    """Establish and validate connection to the MongoDB."""

    client = pymongo.MongoClient(
        mongo_uri, appname="devrel.showcase.anthropic_rag.python"
    )

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

DB_NAME = "knowledge"
COLLECTION_NAME = "research_papers"

db = mongo_client.get_database(DB_NAME)
collection = db.get_collection(COLLECTION_NAME)

collection.delete_many({})

combined_df_json = combined_df.to_dict(orient="records")
collection.insert_many(combined_df_json)

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
                "path": "embedding",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 5,  # Return top 5 matches
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude the _id field
                "embedding": 0,  # Exclude the embedding field
                "score": {
                    "$meta": "vectorSearchScore"  # Include the search score
                },
            }
        },
    ]

    results = collection.aggregate(pipeline)
    return list(results)


# client = anthropic.Client(api_key=ANTHROPIC_API_KEY)

def handle_user_query(query, collection):
    get_knowledge = vector_search(query, collection)

    search_result = ""
    for result in get_knowledge:
        search_result += (
            f"Title: {result.get('title', 'N/A')}, "
            f"Company Name: {result.get('companyName', 'N/A')}, "
            f"Company URL: {result.get('companyUrl', 'N/A')}, "
            f"Date Published: {result.get('published_at', 'N/A')}, "
            f"Article URL: {result.get('url', 'N/A')}, "
            f"Description: {result.get('description', 'N/A')}, \n"
        )

    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        system="You are Venture Captital Tech Analyst with access to some tech company articles and information. You use the information you are given to provide advice.",
        messages=[
            {
                "role": "user",
                "content": "Answer this user query: "
                + query
                + " with the following context: "
                + search_result,
            }
        ],
    )

    return (response.content[0].text), search_result

query = "Give me the best tech stock to invest in and tell me why"
response, source_information = handle_user_query(query, collection)

logger.debug(f"Response: {response}")
logger.debug(f"\nSource Information: \n{source_information}")

logger.info("\n\n[DONE]", bright=True)