from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
StorageContext,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.analyticdb import AnalyticDBVectorStore
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

# AnalyticDB

>[AnalyticDB for PostgreSQL](https://www.alibabacloud.com/help/en/analyticdb-for-postgresql/product-overview/overview-product-overview) is a massively parallel processing (MPP) data warehousing service that is designed to analyze large volumes of data online.


To run this notebook you need a AnalyticDB for PostgreSQL instance running in the cloud (you can get one at [common-buy.aliyun.com](https://common-buy.aliyun.com/?commodityCode=GreenplumPost&regionId=cn-hangzhou&request=%7B%22instance_rs_type%22%3A%22ecs%22%2C%22engine_version%22%3A%226.0%22%2C%22seg_node_num%22%3A%224%22%2C%22SampleData%22%3A%22false%22%2C%22vector_optimizor%22%3A%22Y%22%7D)).

After creating the instance, you should create a manager account by [API](https://www.alibabacloud.com/help/en/analyticdb-for-postgresql/developer-reference/api-gpdb-2016-05-03-createaccount) or 'Account Management' at the instance detail web page.

You should ensure you have `llama-index` installed:
"""
logger.info("# AnalyticDB")

# %pip install llama-index-vector-stores-analyticdb

# !pip install llama-index

"""
### Please provide parameters:
"""
logger.info("### Please provide parameters:")

# import getpass

alibaba_cloud_ak = ""
alibaba_cloud_sk = ""

region_id = "cn-hangzhou"  # region id of the specific instance
instance_id = "gp-xxxx"  # adb instance id
account = "test_account"  # instance account name created by API or 'Account Management' at the instance detail web page
account_password = ""  # instance account password

"""
### Import needed package dependencies:
"""
logger.info("### Import needed package dependencies:")


"""
### Load some example data:
"""
logger.info("### Load some example data:")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

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

analytic_db_store = AnalyticDBVectorStore.from_params(
    access_key_id=alibaba_cloud_ak,
    access_key_secret=alibaba_cloud_sk,
    region_id=region_id,
    instance_id=instance_id,
    account=account,
    account_password=account_password,
    namespace="llama",
    collection="llama",
    metrics="cosine",
    embedding_dimension=1536,
)

"""
### Build the Index from the Documents:
"""
logger.info("### Build the Index from the Documents:")

storage_context = StorageContext.from_defaults(vector_store=analytic_db_store)

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

"""
### Delete the collection:
"""
logger.info("### Delete the collection:")

analytic_db_store.delete_collection()

logger.info("\n\n[DONE]", bright=True)