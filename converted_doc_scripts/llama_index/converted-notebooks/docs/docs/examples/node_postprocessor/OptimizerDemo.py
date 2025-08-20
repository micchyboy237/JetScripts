from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex
from llama_index.core import download_loader
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/OptimizerDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Sentence Embedding Optimizer

This postprocessor optimizes token usage by removing sentences that are not relevant to the query (this is done using embeddings).The percentile cutoff is a measure for using the top percentage of relevant sentences. The threshold cutoff can be specified instead, which uses a raw similarity cutoff for picking which sentences to keep.
"""
logger.info("# Sentence Embedding Optimizer")

# %pip install llama-index-readers-wikipedia


# os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"

"""
### Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("### Setup")

# !pip install llama-index



loader = WikipediaReader()
documents = loader.load_data(pages=["Berlin"])


index = VectorStoreIndex.from_documents(documents)

"""
Compare query with and without optimization for LLM token usage, Embedding Model usage on query, Embedding model usage for optimizer, and total time.
"""
logger.info("Compare query with and without optimization for LLM token usage, Embedding Model usage on query, Embedding model usage for optimizer, and total time.")


logger.debug("Without optimization")
start_time = time.time()
query_engine = index.as_query_engine()
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
logger.debug("Total time elapsed: {}".format(end_time - start_time))
logger.debug("Answer: {}".format(res))

logger.debug("With optimization")
start_time = time.time()
query_engine = index.as_query_engine(
    node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)]
)
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
logger.debug("Total time elapsed: {}".format(end_time - start_time))
logger.debug("Answer: {}".format(res))

logger.debug("Alternate optimization cutoff")
start_time = time.time()
query_engine = index.as_query_engine(
    node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.7)]
)
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
logger.debug("Total time elapsed: {}".format(end_time - start_time))
logger.debug("Answer: {}".format(res))

logger.info("\n\n[DONE]", bright=True)