from jet.models.config import MODELS_CACHE_DIR
from datasets import load_dataset
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from tqdm import tqdm
import json
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
# How To Build An AI Agent With Claude 3.5 Sonnet, LlamaIndex and MongoDB

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/how_to_build_ai_agent_claude_3_5_sonnet_llamaindex_mongodb.ipynb)

## Set Up Libraries
"""
logger.info("# How To Build An AI Agent With Claude 3.5 Sonnet, LlamaIndex and MongoDB")

# !pip install --quiet llama-index  # main llamaindex libary
# !pip install --quiet llama-index-vector-stores-mongodb # mongodb vector database
# !pip install --quiet llama-index-llms-anthropic # anthropic llm provider
# !pip install --quiet llama-index-embeddings-huggingface # ollama embedding provider
# !pip install --quiet pymongo pandas datasets # others

"""
## Set Up Environment Variables
"""
logger.info("## Set Up Environment Variables")



# os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["HF_TOKEN"] = ""
# os.environ["OPENAI_API_KEY"] = ""

"""
## Configure LLMs and Embedding Models
"""
logger.info("## Configure LLMs and Embedding Models")


llm = Ollama(model="claude-3-5-sonnet-20240620")

embed_model = HuggingFaceEmbedding(
    model="mxbai-embed-large",
    dimensions=256,
    embed_batch_size=10,
#     ollama_api_key=os.environ["OPENAI_API_KEY"],
)

Settings.embed_model = embed_model
Settings.llm = llm

"""
## Data Loading
"""
logger.info("## Data Loading")



dataset = load_dataset("MongoDB/airbnb_embeddings", split="train", streaming=True)
dataset = dataset.take(4000)

dataset_df = pd.DataFrame(dataset)

dataset_df.head(5)

dataset_df = dataset_df.drop(columns=["text_embeddings"])

"""
## Data Processing
"""
logger.info("## Data Processing")



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

"""
## Embedding Generation
"""
logger.info("## Embedding Generation")



base_splitter = SentenceSplitter(chunk_size=5000, chunk_overlap=200)

nodes = base_splitter.get_nodes_from_documents(llama_documents)

pbar = tqdm(total=len(nodes), desc="Embedding Progress", unit="node")

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode=MetadataMode.EMBED)
    )
    node.embedding = node_embedding

    pbar.update(1)

pbar.close()

logger.debug("Embedding process completed!")

"""
## MongoDB Vector Database and Connection Setup

MongoDB acts as both an operational and a vector database for the RAG system.
MongoDB Atlas specifically provides a database solution that efficiently stores, queries and retrieves vector embeddings.

Creating a database and collection within MongoDB is made simple with MongoDB Atlas.

1. First, register for a [MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register). For existing users, sign into MongoDB Atlas.
2. [Follow the instructions](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/). Select Atlas UI as the procedure to deploy your first cluster.
3. Create the database: `airbnb`.
4. Within the database ` airbnb`, create the collection ‘listings_reviews’.
5. Create a [vector search index](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure/) named vector_index for the ‘listings_reviews’ collection. This index enables the RAG application to retrieve records as additional context to supplement user queries via vector search. Below is the JSON definition of the data collection vector search index.

Your vector search index created on MongoDB Atlas should look like below:

```
{
  "fields": [
    {
      "numDimensions": 256,
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

    client = pymongo.MongoClient(
        mongo_uri, appname="devrel.showcase.ollama_llamaindex_agent"
    )

    ping_result = client.admin.command("ping")
    if ping_result.get("ok") == 1.0:
        logger.debug("Connection to MongoDB successful")
        return client
    logger.debug("Connection to MongoDB failed")
    return None


mongo_uri = os.environ.get("MONGO_URI")
if not mongo_uri:
    logger.debug("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

DB_NAME = "airbnb"
COLLECTION_NAME = "listings_reviews"

db = mongo_client.get_database(DB_NAME)
collection = db.get_collection(COLLECTION_NAME)

collection.delete_many({})

"""
## Data Ingestion
"""
logger.info("## Data Ingestion")


vector_store = MongoDBAtlasVectorSearch(
    mongo_client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    index_name="vector_index",
)

vector_store.add(nodes)

"""
## Creating Retriver Tool for Agent
"""
logger.info("## Creating Retriver Tool for Agent")


index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="knowledge_base",
        description=(
            "Provides information about Airbnb listings and reviews."
            "Use a detailed plain text question as input to the tool."
        ),
    ),
)

"""
## AI Agent Creation
"""
logger.info("## AI Agent Creation")


agent_worker = FunctionCallingAgentWorker.from_tools(
    [query_engine_tool], llm=llm, verbose=True
)
agent = agent_worker.as_agent()

response = agent.chat("Tell me the best listing for a place in New York")
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)