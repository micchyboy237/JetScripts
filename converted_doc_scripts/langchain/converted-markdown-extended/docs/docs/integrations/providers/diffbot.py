from jet.logger import logger
from langchain_community.document_loaders import DiffbotLoader
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
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
# Diffbot

> [Diffbot](https://docs.diffbot.com/docs) is a suite of ML-based products that make it easy to structure and integrate web data.

## Installation and Setup

[Get a free Diffbot API token](https://app.diffbot.com/get-started/) and [follow these instructions](https://docs.diffbot.com/reference/authentication) to authenticate your requests.

## Document Loader

Diffbot's [Extract API](https://docs.diffbot.com/reference/extract-introduction) is a service that structures and normalizes data from web pages.

Unlike traditional web scraping tools, `Diffbot Extract` doesn't require any rules to read the content on a page. It uses a computer vision model to classify a page into one of 20 possible types, and then transforms raw HTML markup into JSON. The resulting structured JSON follows a consistent [type-based ontology](https://docs.diffbot.com/docs/ontology), which makes it easy to extract data from multiple different web sources with the same schema.

See a [usage example](/docs/integrations/document_loaders/diffbot).
"""
logger.info("# Diffbot")


"""
## Graphs

Diffbot's [Natural Language Processing API](https://www.diffbot.com/products/natural-language/) allows for the extraction of entities, relationships, and semantic meaning from unstructured text data.

See a [usage example](/docs/integrations/graphs/diffbot).
"""
logger.info("## Graphs")


logger.info("\n\n[DONE]", bright=True)