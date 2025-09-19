from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.astra_db import AstraDBVectorStore
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/AstraDBIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Astra DB

>[DataStax Astra DB](https://docs.datastax.com/en/astra/home/astra.html) is a serverless vector-capable database built on Apache Cassandra and accessed through an easy-to-use JSON API.

To run this notebook you need a DataStax Astra DB instance running in the cloud (you can get one for free at [datastax.com](https://astra.datastax.com)).

You should ensure you have `llama-index` and `astrapy` installed:
"""
logger.info("# Astra DB")

# %pip install llama-index-vector-stores-astra-db
# %pip install llama-index-embeddings-huggingface

# !pip install llama-index
# !pip install "astrapy>=1.0"

"""
### Please provide database connection parameters and secrets:
"""
logger.info("### Please provide database connection parameters and secrets:")

# import getpass

api_endpoint = input(
    "\nPlease enter your Database Endpoint URL (e.g. 'https://4bc...datastax.com'):"
)

# token = getpass.getpass(
    "\nPlease enter your 'Database Administrator' Token (e.g. 'AstraCS:...'):"
)

# os.environ["OPENAI_API_KEY"] = getpass.getpass(
    "\nPlease enter your OllamaFunctionCalling API Key (e.g. 'sk-...'):"
)

"""
### Import needed package dependencies:
"""
logger.info("### Import needed package dependencies:")


"""
### Load some example data:
"""
logger.info("### Load some example data:")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Read the data:
"""
logger.info("### Read the data:")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
logger.debug(f"Total documents: {len(documents)}")
logger.debug(f"First document, id: {documents[0].doc_id}")
logger.debug(f"First document, hash: {documents[0].hash}")
logger.debug(
    "First document, text"
    f" ({len(documents[0].text)} characters):\n{'='*20}\n{documents[0].text[:360]} ..."
)

"""
### Create the Astra DB Vector Store object:
"""
logger.info("### Create the Astra DB Vector Store object:")

astra_db_store = AstraDBVectorStore(
    token=token,
    api_endpoint=api_endpoint,
    collection_name="astra_v_table",
    embedding_dimension=1536,
)

"""
### Build the Index from the Documents:
"""
logger.info("### Build the Index from the Documents:")

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

"""
### Query using the index:
"""
logger.info("### Query using the index:")

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")

logger.debug(response.response)

logger.info("\n\n[DONE]", bright=True)