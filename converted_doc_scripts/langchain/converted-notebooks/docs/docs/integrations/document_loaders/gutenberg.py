from jet.logger import logger
from langchain_community.document_loaders import GutenbergLoader
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
# Gutenberg

>[Project Gutenberg](https://www.gutenberg.org/about/) is an online library of free eBooks.

This notebook covers how to load links to `Gutenberg` e-books into a document format that we can use downstream.
"""
logger.info("# Gutenberg")


loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/69972/pg69972.txt")

data = loader.load()

data[0].page_content[:300]

data[0].metadata

logger.info("\n\n[DONE]", bright=True)