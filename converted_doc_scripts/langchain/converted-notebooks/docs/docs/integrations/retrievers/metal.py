from jet.logger import logger
from langchain_community.retrievers import MetalRetriever
from metal_sdk.metal import Metal
import os
import shutil


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
# Metal

>[Metal](https://github.com/getmetal/metal-python) is a managed service for ML Embeddings.

This notebook shows how to use [Metal's](https://docs.getmetal.io/introduction) retriever.

First, you will need to sign up for Metal and get an API key. You can do so [here](https://docs.getmetal.io/misc-create-app)
"""
logger.info("# Metal")

# %pip install --upgrade --quiet  metal_sdk


API_KEY = ""
CLIENT_ID = ""
INDEX_ID = ""

metal = Metal(API_KEY, CLIENT_ID, INDEX_ID)

"""
## Ingest Documents

You only need to do this if you haven't already set up an index
"""
logger.info("## Ingest Documents")

metal.index({"text": "foo1"})
metal.index({"text": "foo"})

"""
## Query

Now that our index is set up, we can set up a retriever and start querying it.
"""
logger.info("## Query")


retriever = MetalRetriever(metal, params={"limit": 2})

retriever.invoke("foo1")

logger.info("\n\n[DONE]", bright=True)