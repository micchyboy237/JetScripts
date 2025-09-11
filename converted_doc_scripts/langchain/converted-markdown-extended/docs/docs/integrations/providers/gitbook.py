from jet.logger import logger
from langchain_community.document_loaders import GitbookLoader
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
# GitBook

>[GitBook](https://docs.gitbook.com/) is a modern documentation platform where teams can document everything from products to internal knowledge bases and APIs.

## Installation and Setup

There isn't any special setup for it.

## Document Loader

See a [usage example](/docs/integrations/document_loaders/gitbook).
"""
logger.info("# GitBook")


logger.info("\n\n[DONE]", bright=True)