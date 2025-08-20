from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_response
from llama_index.storage.docstore.dynamodb import DynamoDBDocumentStore
from llama_index.storage.index_store.dynamodb import DynamoDBIndexStore
from llama_index.vector_stores.dynamodb import DynamoDBVectorStore
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

"""
# Dynamo DB Docstore Demo

This guide shows you how to directly use our `DocumentStore` abstraction backed by DynamoDB. By putting nodes in the docstore, this allows you to define multiple indices over the same underlying docstore, instead of duplicating data across indices.

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/docstore/DynamoDBDocstoreDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Dynamo DB Docstore Demo")

# %pip install llama-index-storage-docstore-dynamodb
# %pip install llama-index-storage-index-store-dynamodb
# %pip install llama-index-vector-stores-dynamodb
# %pip install llama-index-llms-ollama

# !pip install llama-index

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


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

TABLE_NAME = os.environ["DYNAMODB_TABLE_NAME"]


storage_context = StorageContext.from_defaults(
    docstore=DynamoDBDocumentStore.from_table_name(table_name=TABLE_NAME),
    index_store=DynamoDBIndexStore.from_table_name(table_name=TABLE_NAME),
    vector_store=DynamoDBVectorStore.from_table_name(table_name=TABLE_NAME),
)

storage_context.docstore.add_documents(nodes)

"""
#### Define & Add Multiple Indexes

Each index uses the same underlying Node.
"""
logger.info("#### Define & Add Multiple Indexes")

summary_index = SummaryIndex(nodes, storage_context=storage_context)

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

keyword_table_index = SimpleKeywordTableIndex(
    nodes, storage_context=storage_context
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


storage_context = StorageContext.from_defaults(
    docstore=DynamoDBDocumentStore.from_table_name(table_name=TABLE_NAME),
    index_store=DynamoDBIndexStore.from_table_name(table_name=TABLE_NAME),
    vector_store=DynamoDBVectorStore.from_table_name(table_name=TABLE_NAME),
)

summary_index = load_index_from_storage(
    storage_context=storage_context, index_id=list_id
)
keyword_table_index = load_index_from_storage(
    storage_context=storage_context, index_id=keyword_id
)

vector_index = load_index_from_storage(
    storage_context=storage_context, index_id=vector_id
)

"""
#### Test out some Queries
"""
logger.info("#### Test out some Queries")

chatgpt = MLX(temperature=0, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

Settings.llm = chatgpt
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