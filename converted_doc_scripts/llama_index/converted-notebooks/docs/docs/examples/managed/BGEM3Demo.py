from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.indices.managed.bge_m3 import BGEM3Index
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
In this notebook, we are going to show how to use [BGE-M3](https://huggingface.co/BAAI/bge-m3) with LlamaIndex.

BGE-M3 is a hybrid multilingual retrieval model that supports over 100 languages and can handle input lengths of up to 8,192 tokens. The model can perform (i) dense retrieval, (ii) sparse retrieval, and (iii) multi-vector retrieval.

## Getting Started
"""
logger.info("## Getting Started")

# %pip install llama-index-indices-managed-bge-m3

# %pip install llama-index

"""
## Creating BGEM3Index
"""
logger.info("## Creating BGEM3Index")


Settings.chunk_size = 8192

sentences = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
]
documents = [Document(doc_id=i, text=s) for i, s in enumerate(sentences)]

index = BGEM3Index.from_documents(
    documents,
    weights_for_different_modes=[
        0.4,
        0.2,
        0.4,
    ],  # [dense_weight, sparse_weight, multi_vector_weight]
)

"""
## Retrieve relevant documents
"""
logger.info("## Retrieve relevant documents")

retriever = index.as_retriever()
response = retriever.retrieve("What is BGE-M3?")

"""
## RAG with BGE-M3
"""
logger.info("## RAG with BGE-M3")

query_engine = index.as_query_engine()
response = query_engine.query("What is BGE-M3?")

logger.info("\n\n[DONE]", bright=True)