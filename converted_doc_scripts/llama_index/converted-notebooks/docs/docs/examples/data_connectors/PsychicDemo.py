from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import SummaryIndex
from llama_index.readers.psychic import PsychicReader
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/PsychicDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Psychic Reader
Demonstrates the Psychic data connector. Used to query data from many SaaS tools from a single LlamaIndex-compatible API.

## Prerequisites
Connections must first be established from the Psychic dashboard or React hook before documents can be loaded. Refer to https://docs.psychic.dev/ for more info.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Psychic Reader")

# %pip install llama-index-readers-psychic

# !pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


psychic_key = "PSYCHIC_API_KEY"
account_id = "ACCOUNT_ID"
connector_id = "notion"
documents = PsychicReader(psychic_key=psychic_key).load_data(
    connector_id=connector_id, account_id=account_id
)

# os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is Psychic's privacy policy?")
display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)