from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.azure_openai import AzureMLX
from llama_index.vector_stores.azurecosmosmongo import (
AzureCosmosDBMongoDBVectorSearch,
)
import json
import openai
import os
import pymongo
import shutil
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/AzureCosmosDBMongoDBvCoreDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Azure CosmosDB MongoDB Vector Store
In this notebook we are going to show how to use Azure Cosmosdb Mongodb vCore to perform vector searches in LlamaIndex. We will create the embedding using Azure Open AI.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Azure CosmosDB MongoDB Vector Store")

# %pip install llama-index-embeddings-ollama
# %pip install llama-index-vector-stores-azurecosmosmongo
# %pip install llama-index-llms-azure-openai

# !pip install llama-index


"""
### Setup Azure MLX
The first step is to configure the models. They will be used to create embeddings for the documents loaded into the db and for llm completions.
"""
logger.info("### Setup Azure MLX")


llm = AzureMLX(
    model_name=os.getenv("OPENAI_MODEL_COMPLETION"),
    deployment_name=os.getenv("OPENAI_MODEL_COMPLETION"),
    api_base=os.getenv("OPENAI_API_BASE"),
#     api_key=os.getenv("OPENAI_API_KEY"),
    api_type=os.getenv("OPENAI_API_TYPE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0,
)

embed_model = MLXEmbedding(
    model=os.getenv("OPENAI_MODEL_EMBEDDING"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_EMBEDDING"),
    api_base=os.getenv("OPENAI_API_BASE"),
#     api_key=os.getenv("OPENAI_API_KEY"),
    api_type=os.getenv("OPENAI_API_TYPE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)


Settings.llm = llm
Settings.embed_model = embed_model

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Loading documents
Load the documents stored in the `data/paul_graham/` using the SimpleDirectoryReader
"""
logger.info("### Loading documents")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

logger.debug("Document ID:", documents[0].doc_id)

"""
### Create the index
Here we establish the connection to an Azure Cosmosdb mongodb vCore cluster and create an vector search index.
"""
logger.info("### Create the index")


connection_string = os.environ.get("AZURE_COSMOSDB_MONGODB_URI")
mongodb_client = pymongo.MongoClient(connection_string)
store = AzureCosmosDBMongoDBVectorSearch(
    mongodb_client=mongodb_client,
    db_name="demo_vectordb",
    collection_name="paul_graham_essay",
)
storage_context = StorageContext.from_defaults(vector_store=store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Query the index
We can now ask questions using our index.
"""
logger.info("### Query the index")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author love working on?")


logger.debug(textwrap.fill(str(response), 100))

response = query_engine.query("What did he/she do in summer of 2016?")

logger.debug(textwrap.fill(str(response), 100))

logger.info("\n\n[DONE]", bright=True)