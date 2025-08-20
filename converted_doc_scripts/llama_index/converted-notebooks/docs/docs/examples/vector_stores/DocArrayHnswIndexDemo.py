from IPython.display import Markdown, display
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
GPTVectorStoreIndex,
SimpleDirectoryReader,
Document,
)
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.docarray import DocArrayHnswVectorStore
import logging
import os
import os, shutil
import shutil
import sys
import textwrap
import warnings


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/DocArrayHnswIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# DocArray Hnsw Vector Store

[DocArrayHnswVectorStore](https://docs.docarray.org/user_guide/storing/index_hnswlib/) is a lightweight Document Index implementation provided by [DocArray](https://github.com/docarray/docarray) that runs fully locally and is best suited for small- to medium-sized datasets. It stores vectors on disk in hnswlib, and stores all other data in SQLite.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# DocArray Hnsw Vector Store")

# %pip install llama-index-vector-stores-docarray

# !pip install llama-index



warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"




# os.environ["OPENAI_API_KEY"] = "<your openai key>"

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
logger.debug(
    "Document ID:",
    documents[0].doc_id,
    "Document Hash:",
    documents[0].doc_hash,
)

"""
## Initialization and indexing
"""
logger.info("## Initialization and indexing")



vector_store = DocArrayHnswVectorStore(work_dir="hnsw_index")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = GPTVectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
## Querying
"""
logger.info("## Querying")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
logger.debug(textwrap.fill(str(response), 100))

response = query_engine.query("What was a hard moment for the author?")
logger.debug(textwrap.fill(str(response), 100))

"""
## Querying with filters
"""
logger.info("## Querying with filters")


nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
        },
    ),
]



vector_store = DocArrayHnswVectorStore(work_dir="hnsw_filters")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = GPTVectorStoreIndex(nodes, storage_context=storage_context)



filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")


hnsw_dirs = ["hnsw_filters", "hnsw_index"]
for dir in hnsw_dirs:
    if os.path.exists(dir):
        shutil.rmtree(dir)

logger.info("\n\n[DONE]", bright=True)