from datasets import load_dataset
from jet.logger import CustomLogger
from sentence_transformers import SentenceTransformer
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/rag_mongodb_llama3_huggingface_open_source.ipynb) 

# Implementing RAG pipelines with MongoDB and Llama3 and Open models From Hugging Face

This notebook is designed to demonstrate how to integrate and utilize Hugging Face's open-source models, specifically Llama3, with MongoDB to implement Retrieval-Augmented Generation (RAG) pipelines for enhanced question answering capabilities.

The process involves preparing a dataset of arXiv papers, transforming their data for effective retrieval, setting up a MongoDB database with vector search capabilities, and using llama3 model for generating answers based on the retrieved documents.

Key Highlights:
- Usage of Hugging Face open-source models and MongoDB for creating RAG pipelines.
- Steps include dataset preparation, database setup, data ingestion, and query processing.
- Detailed guidance on setting up MongoDB collections and vector search indexes.
- Integration with the Llama3 model from Hugging Face for answering complex queries.

Follow the following instruction to set up a MongoDB database and enable vector search:
1. [Register a free Atlas account](https://account.mongodb.com/account/register?utm_campaign=devrel&utm_source=community&utm_medium=cta&utm_content=GitHub%20Cookbook&utm_term=richmond.alake)
 or sign in to your existing Atlas account.

2. [Follow the instructions](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/)
 (select Atlas UI as the procedure) to deploy your first cluster, which distributes your data across multiple servers for improved performance and redundancy.
 
 ![image.png](attachment:image.png)

3. For a free Cluser, be sure to select "Shared" option when creating your new cluster. See image below for details

![image-2.png](attachment:image-2.png)

4. Create the database: `knowledge_base`, and collection `research_papers`

## Import Libraries

Import libaries into development environment
"""
logger.info("# Implementing RAG pipelines with MongoDB and Llama3 and Open models From Hugging Face")

# !pip install datasets pandas pymongo sentence_transformers
# !pip install -U transformers
# !pip install accelerate

"""
## Dataset Loading and Preparation

Load the dataset from HuggingFace.

Only using the first 100 datapoint for demo purposes.
"""
logger.info("## Dataset Loading and Preparation")



os.environ["HF_TOKEN"] = (
    "place_hugging_face_access_token here"  # Do not use this in production environment, use a .env file instead
)

dataset = load_dataset("MongoDB/subset_arxiv_papers_with_embeddings")

dataset_df = pd.DataFrame(dataset["train"])

dataset_df.head(5)

dataset_df = dataset_df.head(100)

dataset_df = dataset_df.drop(columns=["embedding"])
dataset_df.head(5)

"""
## Generate Embeddings
"""
logger.info("## Generate Embeddings")


embedding_model = SentenceTransformer("thenlper/gte-large")


def get_embedding(text: str) -> list[float]:
    if not text.strip():
        logger.debug("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)

    return embedding.tolist()


dataset_df["embedding"] = dataset_df.apply(
    lambda x: get_embedding(x["title"] + " " + x["authors"] + " " + x["abstract"]),
    axis=1,
)

dataset_df.head()

"""
## Database and Collection Setup

Complete the steps below if not already carried out previously.
Creating a database and collection within MongoDB is made simple with MongoDB Atlas.

1. [Register a free Atlas account](https://account.mongodb.com/account/register?utm_campaign=devrel&utm_source=community&utm_medium=cta&utm_content=GitHub%20Cookbook&utm_term=richmond.alake)
 or sign in to your existing Atlas account.

2. [Follow the instructions](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/)
 (select Atlas UI as the procedure)  to deploy your first cluster.

3. Create the database: `knowledge_base`.
4. Within the database` knowledge_base`, create the following collections: `research_papers`
5. Create a
[vector search index](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure)
 named `vector_index` for the `research_papers` collection. This index enables the RAG application to retrieve records as additional context to supplement user queries via vector search. Below is the JSON definition of the data collection vector search index.


 Below is a snipper of what the vector search index definition should looks like:
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
"""
logger.info("## Database and Collection Setup")



def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(
            mongo_uri, appname="devrel.showcase.rag_llama3_huggingface"
        )
        logger.debug("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        logger.debug(f"Connection failed: {e}")
        return None


mongo_uri = "mongodb...pName=Cluster0"  # Placeholder, replace with your connection string or actual environment variable fetching method.

if not mongo_uri:
    logger.debug("MONGO_URI not set in environment variables.")

mongo_client = get_mongo_client(mongo_uri)

db = mongo_client["knowledge_base"]
collection = db["research_papers"]

collection.delete_many({})

"""
## Data Ingestion
"""
logger.info("## Data Ingestion")

documents = dataset_df.to_dict("records")
collection.insert_many(documents)

logger.debug("Data ingestion into MongoDB completed")

"""
## Vector Search
"""
logger.info("## Vector Search")

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
            "score": {"$meta": "vectorSearchScore"},  # Include the search score
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]

    results = collection.aggregate(pipeline)
    return list(results)

def get_search_result(query, collection):
    get_knowledge = vector_search(query, collection)

    search_result = ""
    for result in get_knowledge:
        search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('abstract', 'N/A')}\n"

    return search_result

"""
## Handling User Queries
"""
logger.info("## Handling User Queries")

query = "Get me papers on Artificial Intelligence?"
source_information = get_search_result(query, collection)
combined_information = f"Query: {query}\nContinue to answer the query by using the Search Results:\n{source_information}."
messages = [
    {"role": "system", "content": "You are a research assitant!"},
    {"role": "user", "content": combined_information},
]
logger.debug(messages)

"""
## Loading and Using Llama3
"""
logger.info("## Loading and Using Llama3")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1] :]
logger.debug(tokenizer.decode(response, skip_special_tokens=True))

logger.info("\n\n[DONE]", bright=True)