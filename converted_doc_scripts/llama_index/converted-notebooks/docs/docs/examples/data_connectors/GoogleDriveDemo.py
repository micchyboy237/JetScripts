from IPython.display import Markdown, display
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SummaryIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.google import GoogleDriveReader
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/GoogleDriveDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Google Drive Reader
Demonstrates our Google Drive data connector

## Prerequisites

Follow [these steps](https://developers.google.com/drive/api/quickstart/python#set_up_your_environment) to setup your environment.
1. Enable the Google Drive API in your GCP project.
1. Configure an OAuth Consent screen for your GCP project.
    * It is fine to make it "External" if you're not in a Google Workspace.
1. Create client credentials for your application (this notebook).
    * Make sure to use "Desktop app" as the application type.
    * Move these client credentials to the directory this notebook is in, and name it "credentials.json".

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Google Drive Reader")

# %pip install llama-index llama-index-readers-google


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


"""
## Choose Folder to Read

You can find a folder ID by navigating to a folder in Google Drive then selecting the last part of the URL.

For example, with this URL: `https://drive.google.com/drive/u/0/folders/abcdefgh12345678` the folder ID is `abcdefgh12345678`
"""
logger.info("## Choose Folder to Read")

folder_id = ["<your_folder_id>"]

documents = GoogleDriveReader().load_data(folder_id=folder_id)

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)