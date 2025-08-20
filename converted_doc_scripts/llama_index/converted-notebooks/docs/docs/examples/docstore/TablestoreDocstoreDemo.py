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
from llama_index.embeddings.dashscope import (
DashScopeEmbedding,
DashScopeTextEmbeddingModels,
DashScopeTextEmbeddingType,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.storage.docstore.tablestore import TablestoreDocumentStore
from llama_index.storage.index_store.tablestore import TablestoreIndexStore
from llama_index.vector_stores.tablestore import TablestoreVectorStore
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
# Tablestore Demo

This guide shows you how to directly use our `DocumentStore` abstraction backed by Tablestore. By putting nodes in the docstore, this allows you to define multiple indices over the same underlying docstore, instead of duplicating data across indices.
"""
logger.info("# Tablestore Demo")

# %pip install llama-index-storage-docstore-tablestore
# %pip install llama-index-storage-index-store-tablestore
# %pip install llama-index-vector-stores-tablestore

# %pip install llama-index-llms-dashscope
# %pip install llama-index-embeddings-dashscope

# %pip install llama-index
# %pip install matplotlib

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


"""
#### Config Tablestore
Next, we use tablestore's docsstore to perform a demo.
"""
logger.info("#### Config Tablestore")

# import getpass

# os.environ["tablestore_end_point"] = getpass.getpass("tablestore end_point:")
# os.environ["tablestore_instance_name"] = getpass.getpass(
    "tablestore instance_name:"
)
# os.environ["tablestore_access_key_id"] = getpass.getpass(
    "tablestore access_key_id:"
)
# os.environ["tablestore_access_key_secret"] = getpass.getpass(
    "tablestore access_key_secret:"
)

"""
#### Config DashScope LLM
Next, we use dashscope's llm to perform a demo.
"""
logger.info("#### Config DashScope LLM")

# import getpass

# os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("DashScope api key:")

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
#### Init Store/Embedding/LLM/StorageContext
"""
logger.info("#### Init Store/Embedding/LLM/StorageContext")


embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,  # default demiension is 1024
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)

dashscope_llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
Settings.llm = dashscope_llm

docstore = TablestoreDocumentStore.from_config(
    endpoint=os.getenv("tablestore_end_point"),
    instance_name=os.getenv("tablestore_instance_name"),
    access_key_id=os.getenv("tablestore_access_key_id"),
    access_key_secret=os.getenv("tablestore_access_key_secret"),
)

index_store = TablestoreIndexStore.from_config(
    endpoint=os.getenv("tablestore_end_point"),
    instance_name=os.getenv("tablestore_instance_name"),
    access_key_id=os.getenv("tablestore_access_key_id"),
    access_key_secret=os.getenv("tablestore_access_key_secret"),
)

vector_store = TablestoreVectorStore(
    endpoint=os.getenv("tablestore_end_point"),
    instance_name=os.getenv("tablestore_instance_name"),
    access_key_id=os.getenv("tablestore_access_key_id"),
    access_key_secret=os.getenv("tablestore_access_key_secret"),
    vector_dimension=1024,  # embedder dimension is 1024
)
vector_store.create_table_if_not_exist()
vector_store.create_search_index_if_not_exist()

storage_context = StorageContext.from_defaults(
    docstore=docstore, index_store=index_store, vector_store=vector_store
)

"""
#### Add to docStore
"""
logger.info("#### Add to docStore")

storage_context.docstore.add_documents(nodes)

"""
#### Define & Add Multiple Indexes

Each index uses the same underlying Node.
"""
logger.info("#### Define & Add Multiple Indexes")

summary_index = SummaryIndex(nodes, storage_context=storage_context)

vector_index = VectorStoreIndex(
    nodes,
    insert_batch_size=20,
    embed_model=embedder,
    storage_context=storage_context,
)

keyword_table_index = SimpleKeywordTableIndex(
    nodes=nodes,
    storage_context=storage_context,
    llm=dashscope_llm,
)

len(storage_context.docstore.docs)

"""
#### Test out saving and loading
"""
logger.info("#### Test out saving and loading")

storage_context.persist()

list_id = summary_index.index_id
vector_id = vector_index.index_id
keyword_id = keyword_table_index.index_id
logger.debug(list_id, vector_id, keyword_id)


storage_context = StorageContext.from_defaults(
    docstore=docstore, index_store=index_store, vector_store=vector_store
)

summary_index = load_index_from_storage(
    storage_context=storage_context,
    index_id=list_id,
)
keyword_table_index = load_index_from_storage(
    llm=dashscope_llm,
    storage_context=storage_context,
    index_id=keyword_id,
)
vector_index = load_index_from_storage(
    insert_batch_size=20,
    embed_model=embedder,
    storage_context=storage_context,
    index_id=vector_id,
)

"""
#### Test out some Queries
"""
logger.info("#### Test out some Queries")

Settings.llm = dashscope_llm
Settings.chunk_size = 1024

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