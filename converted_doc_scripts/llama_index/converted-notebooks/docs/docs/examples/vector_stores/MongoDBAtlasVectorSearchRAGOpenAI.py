from datasets import load_dataset
from google.colab import userdata
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext
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
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# MongoDB Atlas + MLX RAG Example
"""
logger.info("# MongoDB Atlas + MLX RAG Example")

# !pip install llama-index
# !pip install llama-index-vector-stores-mongodb
# !pip install llama-index-embeddings-ollama
# !pip install pymongo
# !pip install datasets
# !pip install pandas

# %env OPENAI_API_KEY=OPENAI_API_KEY


dataset = load_dataset("AIatMongoDB/embedded_movies")

dataset_df = pd.DataFrame(dataset["train"])

dataset_df.head(5)

dataset_df = dataset_df.dropna(subset=["fullplot"])
logger.debug("\nNumber of missing values in each column after removal:")
logger.debug(dataset_df.isnull().sum())

dataset_df = dataset_df.drop(columns=["plot_embedding"])

dataset_df.head(5)


embed_model = MLXEmbedding(model="mxbai-embed-large", dimensions=256)
llm = MLXLlamaIndexLLMAdapter()

Settings.llm = llm
Settings.embed_model = embed_model


documents_json = dataset_df.to_json(orient="records")
documents_list = json.loads(documents_json)

llama_documents = []

for document in documents_list:
    document["writers"] = json.dumps(document["writers"])
    document["languages"] = json.dumps(document["languages"])
    document["genres"] = json.dumps(document["genres"])
    document["cast"] = json.dumps(document["cast"])
    document["directors"] = json.dumps(document["directors"])
    document["countries"] = json.dumps(document["countries"])
    document["imdb"] = json.dumps(document["imdb"])
    document["awards"] = json.dumps(document["awards"])

    llama_document = Document(
        text=document["fullplot"],
        metadata=document,
        excluded_llm_metadata_keys=["fullplot", "metacritic"],
        excluded_embed_metadata_keys=[
            "fullplot",
            "metacritic",
            "poster",
            "num_mflix_comments",
            "runtime",
            "rated",
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


parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(llama_documents)

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

"""
Ensure your databse, collection and vector store index is setup on MongoDB Atlas for the collection or the following step won't work appropriately on MongoDB.


 - For assistance with database cluster setup and obtaining the URI, refer to this [guide](https://www.mongodb.com/docs/guides/atlas/cluster/) for setting up a MongoDB cluster, and this [guide](https://www.mongodb.com/docs/guides/atlas/connection-string/) to get your connection string. 

 - Once you have successfully created a cluster, create the database and collection within the MongoDB Atlas cluster by clicking “+ Create Database”. The database will be named movies, and the collection will be named movies_records.

 - Creating a vector search index within the movies_records collection is essential for efficient document retrieval from MongoDB into our development environment. To achieve this, refer to the official [guide](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/) on vector search index creation.
"""
logger.info("Ensure your databse, collection and vector store index is setup on MongoDB Atlas for the collection or the following step won't work appropriately on MongoDB.")



def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(mongo_uri)
        logger.debug("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        logger.debug(f"Connection failed: {e}")
        return None


mongo_uri = userdata.get("MONGO_URI")
if not mongo_uri:
    logger.debug("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

DB_NAME = "movies"
COLLECTION_NAME = "movies_records"

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

query = "Recommend a romantic movie suitable for the christmas season and justify your selecton"

response = query_engine.query(query)
display_response(response)
pprint.plogger.debug(response.source_nodes)

logger.info("\n\n[DONE]", bright=True)