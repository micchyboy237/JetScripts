from IPython.display import Markdown, display
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SummaryIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.notion import NotionPageReader
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

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/NotionDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Notion Reader
Demonstrates our Notion data connector
"""
logger.info("# Notion Reader")

# %pip install llama-index-readers-notion


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.")

# !pip install llama-index


integration_token = os.getenv("NOTION_INTEGRATION_TOKEN")
page_ids = ["<page_id>"]
documents = NotionPageReader(integration_token=integration_token).load_data(
    page_ids=page_ids
)

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

display(Markdown(f"<b>{response}</b>"))

"""
You can also pass the id of a database to index all the pages in that database:
"""
logger.info("You can also pass the id of a database to index all the pages in that database:")

database_ids = ["<database-id>"]


documents = NotionPageReader(integration_token=integration_token).load_data(
    database_ids=database_ids
)

logger.debug(documents)

index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
display(Markdown(f"<b>{response}</b>"))

"""
To list all databases in your Notion workspace:
"""
logger.info("To list all databases in your Notion workspace:")

reader = NotionPageReader(integration_token=integration_token)
databases = reader.list_databases()
logger.debug(databases)

logger.info("\n\n[DONE]", bright=True)