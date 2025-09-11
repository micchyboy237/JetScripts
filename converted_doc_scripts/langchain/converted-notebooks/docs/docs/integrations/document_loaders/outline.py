from jet.logger import logger
from langchain_outline.document_loaders.outline import OutlineLoader
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
# Outline Document Loader

>[Outline](https://www.getoutline.com/) is an open-source collaborative knowledge base platform designed for team information sharing.

This notebook shows how to obtain langchain Documents from your Outline collections.

## Overview
The [Outline Document Loader](https://github.com/10Pines/langchain-outline) can be used to load Outline collections as LangChain Documents for integration into Retrieval-Augmented Generation (RAG) workflows.

This example demonstrates:

* Setting up a Document Loader to load all documents from an Outline instance.

### Setup
Before starting, ensure you have the following environment variables set:

* OUTLINE_API_KEY: Your API key for authenticating with your Outline instance (https://www.getoutline.com/developers#section/Authentication).
* OUTLINE_INSTANCE_URL: The URL (including protocol) of your Outline instance.
"""
logger.info("# Outline Document Loader")


os.environ["OUTLINE_API_KEY"] = "ol_api_xyz123"
os.environ["OUTLINE_INSTANCE_URL"] = "https://app.getoutline.com"

"""
## Initialization
To initialize the OutlineLoader, you need the following parameters:

* outline_base_url: The URL of your outline instance (or it will be taken from the environment variable).
* outline_api_key: Your API key for authenticating with your Outline instance (or it will be taken from the environment variable).
* outline_collection_id_list: List of collection ids to be retrieved. If None all will be retrieved.
* page_size: Because the Outline API uses paginated results you can configure how many results (documents) per page will be retrieved per API request.  If this is not specified a default will be used.

## Instantiation
"""
logger.info("## Initialization")


loader = OutlineLoader()

loader = OutlineLoader(
    outline_base_url="YOUR_OUTLINE_URL", outline_
)

"""
## Load
To load and return all documents available in the Outline instance
"""
logger.info("## Load")

loader.load()

"""
## Lazy Load
The lazy_load method allows you to iteratively load documents from the Outline collection, yielding each document as it is fetched:
"""
logger.info("## Lazy Load")

loader.lazy_load()

"""
## API reference

For detailed documentation of all `Outline` features and configurations head to the API reference: https://www.getoutline.com/developers
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)