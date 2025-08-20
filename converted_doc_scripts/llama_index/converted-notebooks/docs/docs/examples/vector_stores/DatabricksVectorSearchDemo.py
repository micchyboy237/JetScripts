from databricks.vector_search.client import (
VectorSearchIndex,
VectorSearchClient,
)
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
ServiceContext,
StorageContext,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.databricks import DatabricksVectorSearch
import os
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
# Databricks Vector Search

Databricks Vector Search is a vector database that is built into the Databricks Intelligence Platform and integrated with its governance and productivity tools. Full docs here: https://docs.databricks.com/en/generative-ai/vector-search.html

Install llama-index and databricks-vectorsearch. You must be inside a Databricks runtime to use the Vector Search python client.
"""
logger.info("# Databricks Vector Search")

# %pip install llama-index llama-index-vector-stores-databricks
# %pip install databricks-vectorsearch

"""
Import databricks dependencies
"""
logger.info("Import databricks dependencies")


"""
Import LlamaIndex dependencies
"""
logger.info("Import LlamaIndex dependencies")


"""
Load example data
"""
logger.info("Load example data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
Read the data
"""
logger.info("Read the data")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
logger.debug(f"Total documents: {len(documents)}")
logger.debug(f"First document, id: {documents[0].doc_id}")
logger.debug(f"First document, hash: {documents[0].hash}")
logger.debug(
    "First document, text"
    f" ({len(documents[0].text)} characters):\n{'='*20}\n{documents[0].text[:360]} ..."
)

"""
Create a Databricks Vector Search endpoint which will serve the index
"""
logger.info("Create a Databricks Vector Search endpoint which will serve the index")

client = VectorSearchClient()
client.create_endpoint(
    name="llamaindex_dbx_vector_store_test_endpoint", endpoint_type="STANDARD"
)

"""
Create the Databricks Vector Search index, and build it from the documents
"""
logger.info("Create the Databricks Vector Search index, and build it from the documents")

databricks_index = client.create_direct_access_index(
    endpoint_name="llamaindex_dbx_vector_store_test_endpoint",
    index_name="my_catalog.my_schema.my_test_table",
    primary_key="my_primary_key_name",
    embedding_dimension=1536,  # match the embeddings model dimension you're going to use
    embedding_vector_column="my_embedding_vector_column_name",  # you name this anything you want - it'll be picked up by the LlamaIndex class
    schema={
        "my_primary_key_name": "string",
        "my_embedding_vector_column_name": "array<double>",
        "text": "string",  # one column must match the text_column in the DatabricksVectorSearch instance created below; this will hold the raw node text,
        "doc_id": "string",  # one column must contain the reference document ID (this will be populated by LlamaIndex automatically)
    },
)

databricks_vector_store = DatabricksVectorSearch(
    index=databricks_index,
    text_column="text",
    columns=None,  # YOU MUST ALSO RECORD YOUR METADATA FIELD NAMES HERE
)  # text_column is required for self-managed embeddings
storage_context = StorageContext.from_defaults(
    vector_store=databricks_vector_store
)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
Query the index
"""
logger.info("Query the index")

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")

logger.debug(response.response)

logger.info("\n\n[DONE]", bright=True)