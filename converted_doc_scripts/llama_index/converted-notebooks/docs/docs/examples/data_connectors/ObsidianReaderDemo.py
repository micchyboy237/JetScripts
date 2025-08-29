from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex
from llama_index.readers.obsidian import ObsidianReader
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/ObsidianReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Obsidian Reader

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Obsidian Reader")

# %pip install llama-index-readers-obsidian

# !pip install llama-index

# %env OPENAI_API_KEY=sk-************


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


documents = ObsidianReader(
    "/Users/hursh/vault"
).load_data()  # Returns list of documents

index = VectorStoreIndex.from_documents(
    documents
)  # Initialize index with documents

query_engine = index.as_query_engine()
res = query_engine.query("What is the meaning of life?")

res.response

logger.info("\n\n[DONE]", bright=True)