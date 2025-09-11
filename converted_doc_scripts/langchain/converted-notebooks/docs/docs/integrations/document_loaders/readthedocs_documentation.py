from jet.logger import logger
from langchain_community.document_loaders import ReadTheDocsLoader
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
# ReadTheDocs Documentation

>[Read the Docs](https://readthedocs.org/) is an open-sourced free software documentation hosting platform. It generates documentation written with the `Sphinx` documentation generator.

This notebook covers how to load content from HTML that was generated as part of a `Read-The-Docs` build.

For an example of this in the wild, see [here](https://github.com/langchain-ai/chat-langchain).

This assumes that the HTML has already been scraped into a folder. This can be done by uncommenting and running the following command
"""
logger.info("# ReadTheDocs Documentation")

# %pip install --upgrade --quiet  beautifulsoup4




loader = ReadTheDocsLoader("rtdocs")

docs = loader.load()

logger.info("\n\n[DONE]", bright=True)