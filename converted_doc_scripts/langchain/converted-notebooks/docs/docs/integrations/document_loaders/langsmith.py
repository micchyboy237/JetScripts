from jet.logger import logger
from langchain_core.document_loaders import LangSmithLoader
from langsmith import Client as LangSmithClient
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
---
sidebar_label: LangSmith
---

# LangSmithLoader

This notebook provides a quick overview for getting started with the LangSmith [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all LangSmithLoader features and configurations head to the [API reference](https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.langsmith.LangSmithLoader.html).

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support|
| :--- | :--- | :---: | :---: |  :---: |
| [LangSmithLoader](https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.langsmith.LangSmithLoader.html) | [langchain-core](https://python.langchain.com/api_reference/core/index.html) | ❌ | ❌ | ❌ | 

### Loader features
| Source | Lazy loading | Native async
| :---: | :---: | :---: | 
| LangSmithLoader | ✅ | ❌ | 

## Setup

To access the LangSmith document loader you'll need to install `langchain-core`, create a [LangSmith](https://langsmith.com) account and get an API key.

### Credentials

Sign up at https://langsmith.com and generate an API key. Once you've done this set the LANGSMITH_API_KEY environment variable:
"""
logger.info("# LangSmithLoader")

# import getpass

if not os.environ.get("LANGSMITH_API_KEY"):
#     os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")

"""
If you want to get automated best-in-class tracing, you can also turn on LangSmith tracing:
"""
logger.info("If you want to get automated best-in-class tracing, you can also turn on LangSmith tracing:")



"""
### Installation

Install `langchain-core`:
"""
logger.info("### Installation")

# %pip install -qU langchain-core

"""
### Clone example dataset

For this example, we'll clone and load a public LangSmith dataset. Cloning creates a copy of this dataset on our personal LangSmith account. You can only load datasets that you have a personal copy of.
"""
logger.info("### Clone example dataset")


ls_client = LangSmithClient()

dataset_name = "LangSmith Few Shot Datasets Notebook"
dataset_public_url = (
    "https://smith.langchain.com/public/55658626-124a-4223-af45-07fb774a6212/d"
)

ls_client.clone_public_dataset(dataset_public_url)

"""
## Initialization

Now we can instantiate our document loader and load documents:
"""
logger.info("## Initialization")


loader = LangSmithLoader(
    dataset_name=dataset_name,
    content_key="question",
    limit=50,
)

"""
## Load
"""
logger.info("## Load")

docs = loader.load()
logger.debug(docs[0].page_content)

logger.debug(docs[0].metadata["inputs"])

logger.debug(docs[0].metadata["outputs"])

list(docs[0].metadata.keys())

"""
## Lazy Load
"""
logger.info("## Lazy Load")

page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        break
len(page)

"""
## API reference

For detailed documentation of all LangSmithLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.langsmith.LangSmithLoader.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)