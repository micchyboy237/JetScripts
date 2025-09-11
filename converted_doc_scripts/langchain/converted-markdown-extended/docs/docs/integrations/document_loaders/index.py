from jet.logger import logger
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
import shutil
import { CategoryTable, IndexTable } from "@theme/FeatureTables";


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
sidebar_position: 0
sidebar_class_name: hidden
---

# Document loaders


DocumentLoaders load data into the standard LangChain Document format.

Each DocumentLoader has its own specific parameters, but they can all be invoked in the same way with the .load method.
An example use case is as follows:
"""
logger.info("# Document loaders")


loader = CSVLoader(
    ...  # <-- Integration specific parameters here
)
data = loader.load()

"""
## Webpages

The below document loaders allow you to load webpages.

See this guide for a starting point: [How to: load web pages](/docs/how_to/document_loader_web).

<CategoryTable category="webpage_loaders" />

## PDFs

The below document loaders allow you to load PDF documents.

See this guide for a starting point: [How to: load PDF files](/docs/how_to/document_loader_pdf).

<CategoryTable category="pdf_loaders" />

## Cloud Providers

The below document loaders allow you to load documents from your favorite cloud providers.

<CategoryTable category="cloud_provider_loaders"/>

## Social Platforms

The below document loaders allow you to load documents from different social media platforms.

<CategoryTable category="social_loaders"/>

## Messaging Services

The below document loaders allow you to load data from different messaging platforms.

<CategoryTable category="messaging_loaders"/>

## Productivity tools

The below document loaders allow you to load data from commonly used productivity tools.

<CategoryTable category="productivity_loaders"/>

## Common File Types

The below document loaders allow you to load data from common data formats.

<CategoryTable category="common_loaders" />


## All document loaders

<IndexTable />
"""
logger.info("## Webpages")

logger.info("\n\n[DONE]", bright=True)