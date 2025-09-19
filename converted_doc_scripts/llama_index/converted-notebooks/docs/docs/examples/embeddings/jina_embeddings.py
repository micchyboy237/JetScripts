from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.embeddings.huggingface import (
HuggingFaceEmbedding,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_api import (
HuggingFaceInferenceAPIEmbedding,
)
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Jina 8K Context Window Embeddings

Here we show you how to use `jina-embeddings-v2` which support an 8k context length and is on-par with `text-embedding-ada-002`

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/jina_embeddings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""
logger.info("# Jina 8K Context Window Embeddings")

# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-embeddings-huggingface-api
# %pip install llama-index-embeddings-huggingface

# import nest_asyncio

# nest_asyncio.apply()

"""
## Setup Embedding Model
"""
logger.info("## Setup Embedding Model")


model_name = "jinaai/jina-embeddings-v2-small-en"

embed_model = HuggingFaceEmbedding(
    model_name=model_name, trust_remote_code=True
)

Settings.embed_model = embed_model
Settings.chunk_size = 1024

"""
### Setup OllamaFunctionCalling ada embeddings as comparison
"""
logger.info("### Setup OllamaFunctionCalling ada embeddings as comparison")

embed_model_base = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

"""
## Setup Index to test this out

We'll use our standard Paul Graham example.
"""
logger.info("## Setup Index to test this out")


reader = SimpleDirectoryReader("./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data")
docs = reader.load_data()

index_jina = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

index_base = VectorStoreIndex.from_documents(
    docs, embed_model=embed_model_base
)

"""
## View Results

Look at retrieved results with Jina-8k vs. Replicate
"""
logger.info("## View Results")


retriever_jina = index_jina.as_retriever(similarity_top_k=1)
retriever_base = index_base.as_retriever(similarity_top_k=1)

retrieved_nodes = retriever_jina.retrieve(
    "What did the author do in art school?"
)

for n in retrieved_nodes:
    display_source_node(n, source_length=2000)

retrieved_nodes = retriever_base.retrieve("What did the author do in school?")

for n in retrieved_nodes:
    display_source_node(n, source_length=2000)

logger.info("\n\n[DONE]", bright=True)