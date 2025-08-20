from IPython.display import Markdown, display
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.google import GoogleMapsTextSearchReader
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/GoogleMapsTextSearchReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Google Maps Text Search Reader
This notebook demonstrates how to use the GoogleMapsTextSearchReader from the llama_index library to load and query data from the Google Maps Places API.

If you're opening this Notebook on colab, you will need to install the llama-index library.
"""
logger.info("# Google Maps Text Search Reader")

# !pip install llama-index llama-index-readers-google

"""
### Importing Necessary Libraries
We will import the necessary libraries including the GoogleMapsTextSearchReader from llama_index and other utility libraries.
"""
logger.info("### Importing Necessary Libraries")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
### Setting Up API Key
Make sure you have your Google Maps API key ready. You can set it directly in the code or store it in an environment variable named `GOOGLE_MAPS_API_KEY`.
"""
logger.info("### Setting Up API Key")

os.environ["GOOGLE_MAPS_API_KEY"] = api_key

"""
### Loading Data from Google Maps
Using the `GoogleMapsTextSearchReader`, we will load data for a search query. In this example, we search for quality Turkish food in Istanbul.
"""
logger.info("### Loading Data from Google Maps")

loader = GoogleMapsTextSearchReader()
documents = loader.load_data(
    text="I want to eat quality Turkish food in Istanbul",
    number_of_results=160,
)

logger.debug(documents[0])

"""
### Indexing the Loaded Data
We will now create a VectorStoreIndex from the loaded documents. This index will allow us to perform efficient queries on the data.
"""
logger.info("### Indexing the Loaded Data")

index = VectorStoreIndex.from_documents(documents)

"""
### Querying the Index
Finally, we will query the index to find the Turkish restaurant with the best reviews.
"""
logger.info("### Querying the Index")

response = index.query("Which Turkish restaurant has the best reviews?")
display(Markdown(f"<b>{response}</b>"))

"""
### Summary
In this notebook, we demonstrated how to use the GoogleMapsTextSearchReader to load data from Google Maps, index it using the VectorStoreIndex, and perform a query to find the best-reviewed Turkish restaurant in Istanbul.
"""
logger.info("### Summary")

logger.info("\n\n[DONE]", bright=True)