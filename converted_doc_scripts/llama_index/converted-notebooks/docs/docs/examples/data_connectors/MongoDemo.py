from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import SummaryIndex
from llama_index.readers.mongodb import SimpleMongoReader
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/MongoDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MongoDB Reader
Demonstrates our MongoDB data connector
"""
logger.info("# MongoDB Reader")

# %pip install llama-index-readers-mongodb


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙 and pymongo.
"""
logger.info("If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙 and pymongo.")

# !pip install llama-index pymongo


host = "<host>"
port = "<port>"
db_name = "<db_name>"
collection_name = "<collection_name>"
query_dict = {}
field_names = ["text"]
reader = SimpleMongoReader(host, port)
documents = reader.load_data(
    db_name, collection_name, field_names, query_dict=query_dict
)

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)