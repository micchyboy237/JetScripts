from datasets import load_dataset
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.schema import MetadataMode
from llama_index.core.settings import Settings
from llama_index.embeddings.fireworks import FireworksEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks
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
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# MongoDB Atlas + Fireworks AI RAG Example
"""
logger.info("# MongoDB Atlas + Fireworks AI RAG Example")

# !pip install -q llama-index llama-index-vector-stores-mongodb llama-index-embeddings-fireworks==0.1.2 llama-index-llms-fireworks
# !pip install -q pymongo datasets pandas

# import getpass

# fw_api_key = getpass.getpass("Fireworks API Key:")
os.environ["FIREWORKS_API_KEY"] = fw_api_key


dataset = load_dataset("AIatMongoDB/whatscooking.restaurants")

dataset_df = pd.DataFrame(dataset["train"])

dataset_df.head(5)


embed_model = FireworksEmbedding(
    embed_batch_size=512,
    model_name="nomic-ai/nomic-embed-text-v1.5",
    api_key=fw_api_key,
)
llm = Fireworks(
    temperature=0,
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    api_key=fw_api_key,
)

Settings.llm = llm
Settings.embed_model = embed_model


documents_json = dataset_df.to_json(orient="records")
documents_list = json.loads(documents_json)

llama_documents = []

for document in documents_list:
    document["name"] = json.dumps(document["name"])
    document["cuisine"] = json.dumps(document["cuisine"])
    document["attributes"] = json.dumps(document["attributes"])
    document["menu"] = json.dumps(document["menu"])
    document["borough"] = json.dumps(document["borough"])
    document["address"] = json.dumps(document["address"])
    document["PriceRange"] = json.dumps(document["PriceRange"])
    document["HappyHour"] = json.dumps(document["HappyHour"])
    document["review_count"] = json.dumps(document["review_count"])
    document["TakeOut"] = json.dumps(document["TakeOut"])
    del document["embedding"]
    del document["location"]

    llama_document = Document(
        text=json.dumps(document),
        metadata=document,
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


parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(llama_documents)
new_nodes = nodes[:2500]

node_embeddings = embed_model(new_nodes)

for idx, n in enumerate(new_nodes):
    n.embedding = node_embeddings[idx].embedding
    if "_id" in n.metadata:
        del n.metadata["_id"]

"""
Ensure your database, collection and vector store index is setup on MongoDB Atlas for the collection or the following step won't work appropriately on MongoDB.


 - For assistance with database cluster setup and obtaining the URI, refer to this [guide](https://www.mongodb.com/docs/guides/atlas/cluster/) for setting up a MongoDB cluster, and this [guide](https://www.mongodb.com/docs/guides/atlas/connection-string/) to get your connection string. 

 - Once you have successfully created a cluster, create the database and collection within the MongoDB Atlas cluster by clicking “+ Create Database”. The database will be named movies, and the collection will be named movies_records.

 - Creating a vector search index within the movies_records collection is essential for efficient document retrieval from MongoDB into our development environment. To achieve this, refer to the official [guide](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/) on vector search index creation.
"""
logger.info("Ensure your database, collection and vector store index is setup on MongoDB Atlas for the collection or the following step won't work appropriately on MongoDB.")



def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(mongo_uri)
        logger.debug("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        logger.debug(f"Connection failed: {e}")
        return None


# import getpass

# mongo_uri = getpass.getpass("MONGO_URI:")
if not mongo_uri:
    logger.debug("MONGO_URI not set")

mongo_client = get_mongo_client(mongo_uri)

DB_NAME = "whatscooking"
COLLECTION_NAME = "restaurants"

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

collection.delete_many({})


vector_store = MongoDBAtlasVectorSearch(
    mongo_client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    index_name="vector_index",
)
vector_store.add(new_nodes)

"""
now make sure you create the search index with the right name here
"""
logger.info("now make sure you create the search index with the right name here")


index = VectorStoreIndex.from_vector_store(vector_store)

# %pip install -q matplotlib


query_engine = index.as_query_engine()

query = "search query: Anything that doesn't have alcohol in it"

response = query_engine.query(query)
display_response(response)
pprint.plogger.debug(response.source_nodes)

logger.info("\n\n[DONE]", bright=True)