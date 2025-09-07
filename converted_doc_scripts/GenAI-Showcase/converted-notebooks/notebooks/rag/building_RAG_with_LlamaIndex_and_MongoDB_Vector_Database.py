from jet.models.config import MODELS_CACHE_DIR
from datasets import load_dataset
from google.colab import userdata
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.schema import MetadataMode
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
import json
import os
import pandas as pd
import pprint
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/building_RAG_with_LlamaIndex_and_MongoDB_Vector_Database.ipynb)
"""

# !pip install llama-index
# !pip install llama-index-vector-stores-mongodb
# !pip install llama-index-embeddings-huggingface
# !pip install pymongo
# !pip install datasets
# !pip install pandas


# os.environ["OPENAI_API_KEY"] = "sk..."



dataset = load_dataset("MongoDB/airbnb_embeddings")

dataset_df = pd.DataFrame(dataset["train"])

dataset_df.head(5)

dataset_df = dataset_df.drop(columns=["text_embeddings"])


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
llm = Ollama()

Settings.llm = llm
Settings.embed_model = embed_model



documents_json = dataset_df.to_json(orient="records")

documents_list = json.loads(documents_json)

llama_documents = []

for document in documents_list:
    document["amenities"] = json.dumps(document["amenities"])
    document["images"] = json.dumps(document["images"])
    document["host"] = json.dumps(document["host"])
    document["address"] = json.dumps(document["address"])
    document["availability"] = json.dumps(document["availability"])
    document["review_scores"] = json.dumps(document["review_scores"])
    document["reviews"] = json.dumps(document["reviews"])
    document["image_embeddings"] = json.dumps(document["image_embeddings"])

    llama_document = Document(
        text=document["description"],
        metadata=document,
        excluded_llm_metadata_keys=[
            "_id",
            "transit",
            "minimum_nights",
            "maximum_nights",
            "cancellation_policy",
            "last_scraped",
            "calendar_last_scraped",
            "first_review",
            "last_review",
            "security_deposit",
            "cleaning_fee",
            "guests_included",
            "host",
            "availability",
            "reviews",
            "image_embeddings",
        ],
        excluded_embed_metadata_keys=[
            "_id",
            "transit",
            "minimum_nights",
            "maximum_nights",
            "cancellation_policy",
            "last_scraped",
            "calendar_last_scraped",
            "first_review",
            "last_review",
            "security_deposit",
            "cleaning_fee",
            "guests_included",
            "host",
            "availability",
            "reviews",
            "image_embeddings",
        ],
        metadata_template="{key}=>{value}",
        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
    )

    llama_documents.append(llama_document)

logger.debug(
    "\nThe LLM sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.LLM),
)
logger.debug(
    "\nThe Embedding model sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.EMBED),
)

llama_documents[0]


parser = SentenceSplitter(chunk_size=5000)
nodes = parser.get_nodes_from_documents(llama_documents)

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode=MetadataMode.EMBED)
    )
    node.embedding = node_embedding

"""
### MONGODB VECTOR DATABASE CONNECTION AND SETUP

MongoDB acts as both an operational and a vector database for the RAG system. 
MongoDB Atlas specifically provides a database solution that efficiently stores, queries and retrieves vector embeddings.

Creating a database and collection within MongoDB is made simple with MongoDB Atlas.

1. First, register for a [MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register). For existing users, sign into MongoDB Atlas.
2. [Follow the instructions](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/). Select Atlas UI as the procedure to deploy your first cluster. 
3. Create the database: `airbnb`.
4. Within the database` airbnb`, create the collection ‘listings_reviews’. 
5. Create a [vector search index](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure/) named vector_index for the ‘listings_reviews’ collection. This index enables the RAG application to retrieve records as additional context to supplement user queries via vector search. Below is the JSON definition of the data collection vector search index. 

Follow MongoDB’s [steps to get the connection](https://www.mongodb.com/docs/manual/reference/connection-string/) string from the Atlas UI. After setting up the database and obtaining the Atlas cluster connection URI, securely store the URI within your development environment.

This guide uses Google Colab, which offers a feature for securely storing environment secrets. These secrets can then be accessed within the development environment. Specifically, the line mongo_uri = userdata.get('MONGO_URI') retrieves the URI from the secure storage.
"""
logger.info("### MONGODB VECTOR DATABASE CONNECTION AND SETUP")



def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(
            mongo_uri, appname="devrel.showcase.rag_llamaindex_mongodb"
        )
        logger.debug("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        logger.debug(f"Connection failed: {e}")
        return None


mongo_uri = userdata.get("MONGO_URI")
if not mongo_uri:
    logger.debug("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

DB_NAME = "airbnb"
COLLECTION_NAME = "listings_reviews"

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

collection.delete_many({})


vector_store = MongoDBAtlasVectorSearch(
    mongo_client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    index_name="vector_index",
)
vector_store.add(nodes)


index = VectorStoreIndex.from_vector_store(vector_store)



query_engine = index.as_query_engine(similarity_top_k=3)

query = "I want to stay in a place that's warm and friendly, and not too far from resturants, can you recommend a place? Include a reason as to why you've chosen your selection"

response = query_engine.query(query)
display_response(response)
pprint.plogger.debug(response.source_nodes)

logger.info("\n\n[DONE]", bright=True)