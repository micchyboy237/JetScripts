from dotenv import load_dotenv
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.zep import ZepVectorStore
from uuid import uuid4
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/ZepIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Zep Vector Store

## A long-term memory store for LLM applications

This notebook demonstrates how to use the Zep Vector Store with LlamaIndex.

## About Zep

Zep makes it easy for developers to add relevant documents, chat history memory & rich user data to their LLM app's prompts.

## Note

Zep can automatically embed your documents. The LlamaIndex implementation of the Zep Vector Store utilizes LlamaIndex's embedders to do so.

## Getting Started

**Quick Start Guide:** https://docs.getzep.com/deployment/quickstart/
**GitHub:** https://github.com/getzep/zep

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Zep Vector Store")

# %pip install llama-index-vector-stores-zep

# !pip install llama-index




logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


load_dotenv()

# openai.api_key = os.environ["OPENAI_API_KEY"]


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Create a Zep Vector Store and Index

You can use an existing Zep Collection, or create a new one.
"""
logger.info("## Create a Zep Vector Store and Index")


zep_api_url = "http://localhost:8000"
collection_name = f"graham{uuid4().hex}"

vector_store = ZepVectorStore(
    api_url=zep_api_url,
    collection_name=collection_name,
    embedding_dimensions=1536,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

logger.debug(str(response))

"""
## Querying with Metadata filters
"""
logger.info("## Querying with Metadata filters")


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

collection_name = f"movies{uuid4().hex}"

vector_store = ZepVectorStore(
    api_url=zep_api_url,
    collection_name=collection_name,
    embedding_dimensions=1536,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)


filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

retriever = index.as_retriever(filters=filters)
result = retriever.retrieve("What is inception about?")

for r in result:
    logger.debug("\n", r.node)
    logger.debug("Score:", r.score)

logger.info("\n\n[DONE]", bright=True)