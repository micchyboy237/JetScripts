from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.settings import Settings
from llama_index.embeddings.azure_openai import AzureMLXEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureMLX
from llama_index.storage.docstore.azure import AzureDocumentStore
from llama_index.storage.index_store.azure import AzureIndexStore
from llama_index.storage.kvstore.azure.base import ServiceMode
import logging
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
# Demo: Azure Table Storage as a Docstore

This guide shows you how to use our `AzureDocumentStore` and `AzureIndexStore` abstractions which are backed by Azure Table Storage. By putting nodes in the docstore, this allows you to define multiple indices over the same underlying docstore, instead of duplicating data across indices.

<a href="https://colab.research.google.com/drive/1qtGtyxoIM6rnqxxrTsfixoez8fZy6T2_?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Demo: Azure Table Storage as a Docstore")

# %pip install matplotlib
# %pip install llama-index
# %pip install llama-index-embeddings-azure-openai
# %pip install llama-index-llms-azure-openai
# %pip install llama-index-storage-kvstore-azure
# %pip install llama-index-storage-docstore-azure
# %pip install llama-index-storage-index-store-azure

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)


"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load Documents
"""
logger.info("#### Load Documents")

reader = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/")
documents = reader.load_data()

"""
#### Parse into Nodes
"""
logger.info("#### Parse into Nodes")


nodes = SentenceSplitter().get_nodes_from_documents(documents)

"""
#### Add to Docstore
"""
logger.info("#### Add to Docstore")


"""
The AzureDocumentStore and AzureIndexStore classes provide several helper methods `from_connection_string`, `from_account_and_key`, `from_sas_token`, `from_aad_token`... to simplify connecting to our Azure Table Storage service.
"""
logger.info("The AzureDocumentStore and AzureIndexStore classes provide several helper methods `from_connection_string`, `from_account_and_key`, `from_sas_token`, `from_aad_token`... to simplify connecting to our Azure Table Storage service.")

storage_context = StorageContext.from_defaults(
    docstore=AzureDocumentStore.from_account_and_key(
        "",
        "",
        service_mode=ServiceMode.STORAGE,
    ),
    index_store=AzureIndexStore.from_account_and_key(
        "",
        "",
        service_mode=ServiceMode.STORAGE,
    ),
)

storage_context.docstore.add_documents(nodes)

"""
If we navigate to our Azure Table Storage, we should now be able to see our documents in the table.

# Define our models

In staying with the Azure theme, let's define our Azure MLX embedding and LLM models.
"""
logger.info("# Define our models")

Settings.embed_model = AzureMLXEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key="",
    azure_endpoint="",
    api_version="2024-03-01-preview",
)
Settings.llm = AzureMLX(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    deployment_name="gpt-4",
    api_key="",
    azure_endpoint="",
    api_version="2024-03-01-preview",
)

"""
#### Define Multiple Indexes

Each index uses the same underlying Nodes.
"""
logger.info("#### Define Multiple Indexes")

summary_index = SummaryIndex(nodes, storage_context=storage_context)

"""
We should now be able to see our `summary_index` in Azure Table Storage.
"""
logger.info("We should now be able to see our `summary_index` in Azure Table Storage.")

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
We should now see an entry for our `vector_index` in Azure Table Storage.
"""
logger.info("We should now see an entry for our `vector_index` in Azure Table Storage.")

keyword_table_index = SimpleKeywordTableIndex(
    nodes, storage_context=storage_context
)

"""
We should now see an entry our `keyword_table_index` in Azure Table Storage
"""
logger.info("We should now see an entry our `keyword_table_index` in Azure Table Storage")

len(storage_context.docstore.docs)

"""
#### Test out saving and loading
"""
logger.info("#### Test out saving and loading")

storage_context.persist()

list_id = summary_index.index_id
vector_id = vector_index.index_id
keyword_id = keyword_table_index.index_id


storage_context = StorageContext.from_defaults(
    persist_dir="./storage",
    docstore=AzureDocumentStore.from_account_and_key(
        "",
        "",
        service_mode=ServiceMode.STORAGE,
    ),
    index_store=AzureIndexStore.from_account_and_key(
        "",
        "",
        service_mode=ServiceMode.STORAGE,
    ),
)

summary_index = load_index_from_storage(
    storage_context=storage_context, index_id=list_id
)
vector_index = load_index_from_storage(
    storage_context=storage_context, index_id=vector_id
)
keyword_table_index = load_index_from_storage(
    storage_context=storage_context, index_id=keyword_id
)

"""
#### Test out some Queries
"""
logger.info("#### Test out some Queries")

query_engine = summary_index.as_query_engine()
list_response = query_engine.query("What is a summary of this document?")

display_response(list_response)

query_engine = vector_index.as_query_engine()
vector_response = query_engine.query("What did the author do growing up?")

display_response(vector_response)

query_engine = keyword_table_index.as_query_engine()
keyword_response = query_engine.query(
    "What did the author do after his time at YC?"
)

display_response(keyword_response)

logger.info("\n\n[DONE]", bright=True)