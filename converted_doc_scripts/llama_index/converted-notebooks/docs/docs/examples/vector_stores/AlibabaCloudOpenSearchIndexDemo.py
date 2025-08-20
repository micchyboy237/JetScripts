from IPython.display import Markdown, display
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.alibabacloud_opensearch import (
AlibabaCloudOpenSearchStore,
AlibabaCloudOpenSearchConfig,
)
import logging
import openai
import os
import shutil
import sys


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/AlibabaCloudOpenSearchIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<a href="https://gallery.pai-ml.com/#/import/https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/AlibabaCloudOpenSearchIndexDemo.ipynb" target="_parent"><img src="https://gallery.pai-ml.com/assets/open-in-dsw.svg" alt="Open in PAI-DSW"/></a>

# Alibaba Cloud OpenSearch Vector Store

>[Alibaba Cloud OpenSearch Vector Search Edition](https://help.aliyun.com/zh/open-search/vector-search-edition/product-overview) is a large-scale distributed search engine that is developed by Alibaba Group. Alibaba Cloud OpenSearch Vector Search Edition provides search services for the entire Alibaba Group, including Taobao, Tmall, Cainiao, Youku, and other e-commerce platforms that are provided for customers in regions outside the Chinese mainland. Alibaba Cloud OpenSearch Vector Search Edition is also a base engine of Alibaba Cloud OpenSearch. After years of development, Alibaba Cloud OpenSearch Vector Search Edition has met the business requirements for high availability, high timeliness, and cost-effectiveness. Alibaba Cloud OpenSearch Vector Search Edition also provides an automated O&M system on which you can build a custom search service based on your business features.

To run, you should have a instance.

### Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Alibaba Cloud OpenSearch Vector Store")

# %pip install llama-index-vector-stores-alibabacloud-opensearch

# %pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
### Please provide MLX access key

In order use embeddings by MLX you need to supply an MLX API Key:
"""
logger.info("### Please provide MLX access key")


# OPENAI_API_KEY = getpass.getpass("MLX API Key:")
# openai.api_key = OPENAI_API_KEY

"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load documents
"""
logger.info("#### Load documents")


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
logger.debug(f"Total documents: {len(documents)}")

"""
### Create the Alibaba Cloud OpenSearch Vector Store object:

To run the next step, you should have a Alibaba Cloud OpenSearch Vector Service instance, and configure a table.
"""
logger.info("### Create the Alibaba Cloud OpenSearch Vector Store object:")

# import nest_asyncio

# nest_asyncio.apply()


config = AlibabaCloudOpenSearchConfig(
    endpoint="*****",
    instance_id="*****",
    username="your_username",
    password="your_password",
    table_name="llama",
)

vector_store = AlibabaCloudOpenSearchStore(config)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
#### Query Index
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

"""
### Connecting to an existing store

Since this store is backed by Alibaba Cloud OpenSearch, it is persistent by definition. So, if you want to connect to a store that was created and populated previously, here is how:
"""
logger.info("### Connecting to an existing store")


config = AlibabaCloudOpenSearchConfig(
    endpoint="***",
    instance_id="***",
    username="your_username",
    password="your_password",
    table_name="llama",
)

vector_store = AlibabaCloudOpenSearchStore(config)

index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author study prior to working on AI?"
)

display(Markdown(f"<b>{response}</b>"))

"""
### Metadata filtering

The Alibaba Cloud OpenSearch vector store support metadata filtering at query time. The following cells, which work on a brand new table, demonstrate this feature.

In this demo, for the sake of brevity, a single source document is loaded (the `./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/paul_graham_essay.txt` text file). Nevertheless, you will attach some custom metadata to the document to illustrate how you can can restrict queries with conditions on the metadata attached to the documents.
"""
logger.info("### Metadata filtering")


config = AlibabaCloudOpenSearchConfig(
    endpoint="****",
    instance_id="****",
    username="your_username",
    password="your_password",
    table_name="llama",
)

md_storage_context = StorageContext.from_defaults(
    vector_store=AlibabaCloudOpenSearchStore(config)
)


def my_file_metadata(file_name: str):
    """Depending on the input file name, associate a different metadata."""
    if "essay" in file_name:
        source_type = "essay"
    elif "dinosaur" in file_name:
        source_type = "dinos"
    else:
        source_type = "other"
    return {"source_type": source_type}


md_documents = SimpleDirectoryReader(
    "./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data", file_metadata=my_file_metadata
).load_data()
md_index = VectorStoreIndex.from_documents(
    md_documents, storage_context=md_storage_context
)

"""
Add filter to query engine:
"""
logger.info("Add filter to query engine:")


md_query_engine = md_index.as_query_engine(
    filters=MetadataFilters(
        filters=[MetadataFilter(key="source_type", value="essay")]
    )
)
md_response = md_query_engine.query(
    "How long it took the author to write his thesis?"
)

display(Markdown(f"<b>{md_response}</b>"))

"""
To test that the filtering is at play, try to change it to use only `"dinos"` documents... there will be no answer this time :)
"""
logger.info("To test that the filtering is at play, try to change it to use only `"dinos"` documents... there will be no answer this time :)")

logger.info("\n\n[DONE]", bright=True)