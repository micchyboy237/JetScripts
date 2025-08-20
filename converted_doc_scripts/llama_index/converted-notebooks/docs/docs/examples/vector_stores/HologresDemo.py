from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
StorageContext,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.hologres import HologresVectorStore
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/AnalyticDBDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Hologres

>[Hologres](https://www.alibabacloud.com/help/en/hologres/) is a one-stop real-time data warehouse, which can support high performance OLAP analysis and high QPS online services.


To run this notebook you need a Hologres instance running in the cloud. You can get one following [this link](https://www.alibabacloud.com/help/en/hologres/getting-started/purchase-a-hologres-instance#task-1918224).

After creating the instance, you should be able to figure out following configurations with [Hologres console](https://www.alibabacloud.com/help/en/hologres/user-guide/instance-list?spm=a2c63.p38356.0.0.79b34766nhwskN)
"""
logger.info("# Hologres")

test_hologres_config = {
    "host": "<host>",
    "port": 80,
    "user": "<user>",
    "password": "<password>",
    "database": "<database>",
    "table_name": "<table_name>",
}

"""
By the way, you need to ensure you have `llama-index` installed:
"""
logger.info("By the way, you need to ensure you have `llama-index` installed:")

# %pip install llama-index-vector-stores-hologres

# !pip install llama-index

"""
### Import needed package dependencies:
"""
logger.info("### Import needed package dependencies:")


"""
### Load some example data:
"""
logger.info("### Load some example data:")

# !mkdir -p 'data/paul_graham/'
# !curl 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -o 'data/paul_graham/paul_graham_essay.txt'

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
### Create the AnalyticDB Vector Store object:
"""
logger.info("### Create the AnalyticDB Vector Store object:")

hologres_store = HologresVectorStore.from_param(
    host=test_hologres_config["host"],
    port=test_hologres_config["port"],
    user=test_hologres_config["user"],
    password=test_hologres_config["password"],
    database=test_hologres_config["database"],
    table_name=test_hologres_config["table_name"],
    embedding_dimension=1536,
    pre_delete_table=True,
)

"""
### Build the Index from the Documents:
"""
logger.info("### Build the Index from the Documents:")

storage_context = StorageContext.from_defaults(vector_store=hologres_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Query using the index:
"""
logger.info("### Query using the index:")

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")

logger.debug(response.response)

logger.info("\n\n[DONE]", bright=True)