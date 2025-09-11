from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import AtlasDB
from langchain_text_splitters import SpacyTextSplitter
import os
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Atlas


>[Atlas](https://docs.nomic.ai/index.html) is a platform by Nomic made for interacting with both small and internet scale unstructured datasets. It enables anyone to visualize, search, and share massive datasets in their browser.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows you how to use functionality related to the `AtlasDB` vectorstore.
"""
logger.info("# Atlas")

# %pip install --upgrade --quiet  spacy

# !python3 -m spacy download en_core_web_sm

# %pip install --upgrade --quiet  nomic

"""
### Load Packages
"""
logger.info("### Load Packages")



ATLAS_TEST_API_KEY = "7xDPkYXSYDc1_ErdTPIcoAR9RNd8YDlkS3nVNXcVoIMZ6"

"""
### Prepare the Data
"""
logger.info("### Prepare the Data")

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = SpacyTextSplitter(separator="|")
texts = []
for doc in text_splitter.split_documents(documents):
    texts.extend(doc.page_content.split("|"))

texts = [e.strip() for e in texts]

"""
### Map the Data using Nomic's Atlas
"""
logger.info("### Map the Data using Nomic's Atlas")

db = AtlasDB.from_texts(
    texts=texts,
    name="test_index_" + str(time.time()),  # unique name for your vector store
    description="test_index",  # a description for your vector store
    api_key=ATLAS_TEST_API_KEY,
    index_kwargs={"build_topic_model": True},
)

db.project.wait_for_project_lock()

db.project

"""
Here is a map with the result of this code. This map displays the texts of the State of the Union.
https://atlas.nomic.ai/map/3e4de075-89ff-486a-845c-36c23f30bb67/d8ce2284-8edb-4050-8b9b-9bb543d7f647
"""
logger.info("Here is a map with the result of this code. This map displays the texts of the State of the Union.")

logger.info("\n\n[DONE]", bright=True)