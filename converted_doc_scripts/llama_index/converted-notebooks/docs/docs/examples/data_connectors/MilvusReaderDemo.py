from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.milvus import MilvusReader
import logging
import os
import random
import shutil
import sys
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/MilvusReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MilvusReader

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# MilvusReader")

# %pip install llama-index-readers-milvus

# !pip install llama-index





# os.environ["OPENAI_API_KEY"] = "sk-"

reader = MilvusReader()
reader.load_data([random.random() for _ in range(1536)], "llamalection")

logger.info("\n\n[DONE]", bright=True)