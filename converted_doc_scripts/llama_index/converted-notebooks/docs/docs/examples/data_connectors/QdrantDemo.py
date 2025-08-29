from jet.logger import CustomLogger
from llama_index.readers.qdrant import QdrantReader
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/QdrantDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Qdrant Reader
"""
logger.info("# Qdrant Reader")

# %pip install llama-index-readers-qdrant


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.")

# !pip install llama-index


reader = QdrantReader(host="localhost")

query_vector = [n1, n2, n3, ...]

documents = reader.load_data(
    collection_name="demo", query_vector=query_vector, limit=5
)

"""
### Create index
"""
logger.info("### Create index")

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)