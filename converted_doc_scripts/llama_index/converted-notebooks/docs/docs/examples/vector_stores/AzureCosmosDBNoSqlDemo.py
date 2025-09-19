from jet.models.config import MODELS_CACHE_DIR
from azure.cosmos import CosmosClient, PartitionKey
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.azure_openai import AzureHuggingFaceEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOllamaFunctionCallingAdapter
from llama_index.vector_stores.azurecosmosnosql import (
AzureCosmosDBNoSqlVectorSearch,
)
import json
import openai
import os
import shutil
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Azure Cosmos DB No SQL Vector Store

In this notebook we are going to show a quick demo of how to use AzureCosmosDBNoSqlVectorSearch to perform vector searches in LlamaIndex.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Azure Cosmos DB No SQL Vector Store")

# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-llms-azure-openai

# !pip install llama-index


"""
# Setup Azure OllamaFunctionCalling

The first step is to configure the llm and the embeddings model. These models will be used to create embeddings for the documents loaded into the database and for llm completions.
"""
logger.info("# Setup Azure OllamaFunctionCalling")

llm = AzureOllamaFunctionCallingAdapter(
    model="AZURE_OPENAI_MODEL",
    deployment_name="AZURE_OPENAI_DEPLOYMENT_NAME",
    azure_endpoint="AZURE_OPENAI_BASE",
    api_key="AZURE_OPENAI_KEY",
    api_version="AZURE_OPENAI_VERSION",
)

embed_model = AzureHuggingFaceEmbedding(
    model="AZURE_OPENAI_EMBEDDING_MODEL",
    deployment_name="AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME",
    azure_endpoint="AZURE_OPENAI_BASE",
    api_key="AZURE_OPENAI_KEY",
    api_version="AZURE_OPENAI_VERSION",
)


Settings.llm = llm
Settings.embed_model = embed_model

"""
# Loading Documents

In this example we will be using the paul_graham essay which will be processed by the SimpleDirectoryReader.
"""
logger.info("# Loading Documents")


documents = SimpleDirectoryReader(
    input_files=[r"\docs\examples\data\paul_graham\paul_graham_essay.txt"]
).load_data()

logger.debug("Document ID:", documents[0].doc_id)

"""
# Create the index

Here we establish the connection to cosmos db nosql and create a vector store index.
"""
logger.info("# Create the index")


URI = "AZURE_COSMOSDB_URI"
KEY = "AZURE_COSMOSDB_KEY"
client = CosmosClient(URI, credential=KEY)

indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
}

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 3072,
        }
    ]
}

partition_key = PartitionKey(path="/id")
cosmos_container_properties_test = {"partition_key": partition_key}
cosmos_database_properties_test = {}

store = AzureCosmosDBNoSqlVectorSearch(
    cosmos_client=client,
    vector_embedding_policy=vector_embedding_policy,
    indexing_policy=indexing_policy,
    cosmos_container_properties=cosmos_container_properties_test,
    cosmos_database_properties=cosmos_database_properties_test,
    create_container=True,
)

storage_context = StorageContext.from_defaults(vector_store=store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
# Query the index
We can now ask questions using our index.
"""
logger.info("# Query the index")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author love working on?")


logger.debug(textwrap.fill(str(response), 100))

logger.info("\n\n[DONE]", bright=True)