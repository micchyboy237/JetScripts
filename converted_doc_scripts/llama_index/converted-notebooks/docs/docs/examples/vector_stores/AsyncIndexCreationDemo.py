from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex, download_loader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
import os
import shutil
import time


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/AsyncIndexCreationDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Simple Vector Store - Async Index Creation

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Simple Vector Store - Async Index Creation")

# %pip install llama-index-readers-wikipedia

# !pip install llama-index


# import nest_asyncio

# nest_asyncio.apply()


# os.environ["OPENAI_API_KEY"] = "[YOUR_API_KEY]"



loader = WikipediaReader()
documents = loader.load_data(
    pages=[
        "Berlin",
        "Santiago",
        "Moscow",
        "Tokyo",
        "Jakarta",
        "Cairo",
        "Bogota",
        "Shanghai",
        "Damascus",
    ]
)

len(documents)

"""
9 Wikipedia articles downloaded as documents
"""
logger.info("9 Wikipedia articles downloaded as documents")

start_time = time.perf_counter()
index = VectorStoreIndex.from_documents(documents)
duration = time.perf_counter() - start_time
logger.debug(duration)

"""
Standard index creation took 7.69 seconds
"""
logger.info("Standard index creation took 7.69 seconds")

start_time = time.perf_counter()
index = VectorStoreIndex(documents, use_async=True)
duration = time.perf_counter() - start_time
logger.debug(duration)

"""
Async index creation took 2.37 seconds
"""
logger.info("Async index creation took 2.37 seconds")

query_engine = index.as_query_engine()
query_engine.query("What is the etymology of Jakarta?")

logger.info("\n\n[DONE]", bright=True)