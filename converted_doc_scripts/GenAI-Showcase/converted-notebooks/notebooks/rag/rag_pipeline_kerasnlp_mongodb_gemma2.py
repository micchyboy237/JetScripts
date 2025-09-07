from IPython.display import Markdown
from datasets import load_dataset
from google.colab import userdata
from jet.logger import CustomLogger
from tqdm.notebook import tqdm
from typing import Dict, Optional
import keras
import keras_nlp
import ollama
import os
import pandas as pd
import pymongo
import shutil
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# RAG Pipeline With Keras NLP, MongoDB and Ollama

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/rag_pipeline_kerasnlp_mongodb_gemma2.ipynb)

## Set Up Libraries
"""
logger.info("# RAG Pipeline With Keras NLP, MongoDB and Ollama")

# !pip --quiet install keras
# !pip --quiet install keras-nlp
# !pip --quiet install --upgrade --quiet datasets pandas pymongo
# !pip --quiet install ollama

"""
## Set Up Environment Variables
"""
logger.info("## Set Up Environment Variables")


os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch".
# os.environ["OPENAI_API_KEY"] = ""

"""
## Data Loading
"""
logger.info("## Data Loading")



dataset = load_dataset(
    "MongoDB/subset_arxiv_papers_with_embeddings", split="train", streaming=True
)
dataset = dataset.take(4000)

dataset_df = pd.DataFrame(dataset)

dataset_df.head(5)

"""
## Data Cleaning
"""
logger.info("## Data Cleaning")

dataset_df = dataset_df.dropna(subset=["abstract", "title"])

dataset_df = dataset_df.drop(columns=["embedding"])

"""
## Embedding Generation
"""
logger.info("## Embedding Generation")


# ollama.api_key = os.environ["OPENAI_API_KEY"]

EMBEDDING_MODEL = "mxbai-embed-large"


def get_embedding(text):
    """Generate an embedding for the given text using Ollama's API."""
    if not text or not isinstance(text, str):
        return None

    try:
        embedding = (
            ollama.embeddings.create(input=text, model=EMBEDDING_MODEL, dimensions=1536)
            .data[0]
            .embedding
        )
        return embedding
    except Exception as e:
        logger.debug(f"Error in get_embedding: {e}")
        return None


def combine_columns(row, columns):
    """Combine the contents of specified columns into a single string."""
    return " ".join(str(row[col]) for col in columns if pd.notna(row[col]))


def apply_embedding_with_progress(df, columns):
    """Apply embedding to concatenated text from multiple dataframe columns with a progress bar."""
    if not all(col in df.columns for col in columns):
        missing_cols = [col for col in columns if col not in df.columns]
        raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")

    tqdm.pandas(desc=f"Generating embeddings for columns: {', '.join(columns)}")

    df["combined_text"] = df.apply(lambda row: combine_columns(row, columns), axis=1)

    df["embedding"] = df["combined_text"].progress_apply(get_embedding)

    df = df.drop(columns=["combined_text"])

    return df


try:
    dataset_df = dataset_df.drop(columns=["embedding"], errors="ignore")

    columns_to_embed = [
        "abstract",
        "title",
    ]  # Add or remove columns as needed (text only)
    dataset_df = apply_embedding_with_progress(dataset_df, columns_to_embed)
except Exception as e:
    logger.debug(f"An error occurred: {e}")

dataset_df[columns_to_embed + ["embedding"]].head()

"""
## MongoDB Vector Database and Connection Setup

MongoDB acts as both an operational and a vector database for the RAG system.
MongoDB Atlas specifically provides a database solution that efficiently stores, queries and retrieves vector embeddings.

Creating a database and collection within MongoDB is made simple with MongoDB Atlas.

1. First, register for a [MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register). For existing users, sign into MongoDB Atlas.
2. [Follow the instructions](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/). Select Atlas UI as the procedure to deploy your first cluster.
3. Create the database: `knowledge`.
4. Within the database ` research_papers`, create the collection ‘listings_reviews’.
5. Create a [vector search index](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure/) named vector_index for the ‘listings_reviews’ collection. This index enables the RAG application to retrieve records as additional context to supplement user queries via vector search. Below is the JSON definition of the data collection vector search index.

Your vector search index created on MongoDB Atlas should look like below:

```
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}

```

Follow MongoDB’s [steps to get the connection](https://www.mongodb.com/docs/manual/reference/connection-string/) string from the Atlas UI. After setting up the database and obtaining the Atlas cluster connection URI, securely store the URI within your development environment.

This guide uses Google Colab, which offers a feature for securely storing environment secrets. These secrets can then be accessed within the development environment. Specifically, the line mongo_uri = userdata.get('MONGO_URI') retrieves the URI from the secure storage.
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

DB_NAME = "knowledge"
COLLECTION_NAME = "research_papers"

db = mongo_client.get_database(DB_NAME)
collection = db.get_collection(COLLECTION_NAME)

collection.delete_many({})

"""
## Data Ingestion
"""
logger.info("## Data Ingestion")

try:
    collection.insert_many(dataset_df.to_dict("records"))
    logger.debug("Data ingestion into MongoDB completed")
except Exception as e:
    logger.debug(f"An error occurred during data ingestion: {e}")

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

    pipeline = [vector_search_stage, project_stage]

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

query = "Give me a recommended paper on machine learning"
source_information = get_search_result(query, collection)
combined_information = f"Query: {query}\nContinue to answer the query by using the Search Results:\n{source_information}."

logger.debug(combined_information)

"""
## Keras Config and Markdown
"""
logger.info("## Keras Config and Markdown")



keras.config.set_floatx("bfloat16")


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

"""
## Handle Response Generation and History
"""
logger.info("## Handle Response Generation and History")



class GemmaChat:
    __START_TURN__ = "<start_of_turn>"
    __END_TURN__ = "<end_of_turn>"
    __SYSTEM_STOP__ = "<eos>"

    def __init__(
        self, model, system: str = "", history: Optional[Dict[str, str]] = None
    ):
        self.model = model
        self.system = system
        self.history_params = history or {}
        self.client = pymongo.MongoClient(
            self.history_params.get("connection_string", "mongodb://localhost:27017/")
        )
        self.db = self.client[self.history_params.get("database", "gemma_chat")]
        self.collection = self.db[self.history_params.get("collection", "chat_history")]
        self.session_id = self.history_params.get("session_id", "default_session")

    def format_message(self, message: str, prefix: str = "") -> str:
        return f"{self.__START_TURN__}{prefix}\n{message}{self.__END_TURN__}\n"

    def add_to_history(self, message: str, prefix: str = ""):
        formatted_message = self.format_message(message, prefix)
        self.collection.insert_one(
            {"session_id": self.session_id, "message": formatted_message}
        )

    def get_full_prompt(self) -> str:
        history = self.collection.find({"session_id": self.session_id}).sort("_id", 1)
        prompt = self.system + "\n" + "\n".join([item["message"] for item in history])
        return prompt

    def send_message(self, message: str) -> str:
        self.add_to_history(message, "user")
        prompt = self.get_full_prompt()
        response = self.model.generate(prompt, max_length=2048)
        result = response.replace(prompt, "").replace(self.__SYSTEM_STOP__, "")
        self.add_to_history(result, "model")
        return result

    def show_history(self):
        history = self.collection.find({"session_id": self.session_id}).sort("_id", 1)
        for item in history:
            logger.debug(item["message"])

"""
## Gemma2 Model Initalisation
"""
logger.info("## Gemma2 Model Initalisation")

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(
    "hf://gg-tt/gemma-2-instruct-9b-keras"
)
gemma_lm.summary()

# %time result = gemma_lm.generate("What are your current capabilities?", max_length=256)
to_markdown(result)  # noqa: F821

"""
## Query Gemma 2 with Retrieved Data
"""
logger.info("## Query Gemma 2 with Retrieved Data")

history_params = {
    "connection_string": userdata.get("MONGO_URI"),
    "database": DB_NAME,
    "collection": "chat_history",
    "session_id": "unique_session_id",
}

gemma_chat = GemmaChat(
    gemma_lm, system="You are a research assistant", history=history_params
)

result = gemma_chat.send_message(combined_information)
to_markdown(result)

"""
## View Chat History
"""
logger.info("## View Chat History")

gemma_chat.show_history()

logger.info("\n\n[DONE]", bright=True)