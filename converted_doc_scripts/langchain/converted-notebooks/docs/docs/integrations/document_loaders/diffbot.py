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

>[Diffbot](https://docs.diffbot.com/docs/getting-started-with-diffbot) is a suite of ML-based products that make it easy to structure web data.

>Diffbot's [Extract API](https://docs.diffbot.com/reference/extract-introduction) is a service that structures and normalizes data from web pages.

>Unlike traditional web scraping tools, `Diffbot Extract` doesn't require any rules to read the content on a page. It uses a computer vision model to classify a page into one of 20 possible types, and then transforms raw HTML markup into JSON. The resulting structured JSON follows a consistent [type-based ontology](https://docs.diffbot.com/docs/ontology), which makes it easy to extract data from multiple different web sources with the same schema.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/integrations/document_loaders/diffbot.ipynb)

## Overview
This guide covers how to extract data from a list of URLs using the [Diffbot Extract API](https://www.diffbot.com/products/extract/) into structured JSON that we can use downstream.

## Setting up

Start by installing the required packages.
"""
logger.info("# Diffbot")

# %pip install --upgrade --quiet langchain-community

"""
Diffbot's Extract API requires an API token. Follow these instructions to [get a free API token](/docs/integrations/providers/diffbot#installation-and-setup) and then set an environment variable.
"""
logger.info("Diffbot's Extract API requires an API token. Follow these instructions to [get a free API token](/docs/integrations/providers/diffbot#installation-and-setup) and then set an environment variable.")

# %env DIFFBOT_API_TOKEN REPLACE_WITH_YOUR_TOKEN

"""
## Using the Document Loader

Import the DiffbotLoader module and instantiate it with a list of URLs and your Diffbot token.
"""
logger.info("## Using the Document Loader")



urls = [
    "https://python.langchain.com/",
]

loader = DiffbotLoader(urls=urls, api_token=os.environ.get("DIFFBOT_API_TOKEN"))

"""
With the `.load()` method, you can see the documents loaded
"""
logger.info("With the `.load()` method, you can see the documents loaded")

loader.load()

"""
## Transform Extracted Text to a Graph Document

Structured page content can be further processed with `DiffbotGraphTransformer` to extract entities and relationships into a graph.
"""
logger.info("## Transform Extracted Text to a Graph Document")

# %pip install --upgrade --quiet langchain-experimental


diffbot_nlp = DiffbotGraphTransformer(
    diffbot_api_key=os.environ.get("DIFFBOT_API_TOKEN")
)
graph_documents = diffbot_nlp.convert_to_graph_documents(loader.load())

"""
To continue loading the data into a Knowledge Graph, follow the [`DiffbotGraphTransformer` guide](/docs/integrations/graphs/diffbot/#loading-the-data-into-a-knowledge-graph).
"""
logger.info("To continue loading the data into a Knowledge Graph, follow the [`DiffbotGraphTransformer` guide](/docs/integrations/graphs/diffbot/#loading-the-data-into-a-knowledge-graph).")

logger.info("\n\n[DONE]", bright=True)