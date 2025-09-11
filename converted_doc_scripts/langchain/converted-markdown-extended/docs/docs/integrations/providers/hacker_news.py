from jet.logger import logger
from langchain_community.document_loaders import HNLoader
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
# Hacker News

>[Hacker News](https://en.wikipedia.org/wiki/Hacker_News) (sometimes abbreviated as `HN`) is a social news
> website focusing on computer science and entrepreneurship. It is run by the investment fund and startup
> incubator `Y Combinator`. In general, content that can be submitted is defined as "anything that gratifies
> one's intellectual curiosity."

## Installation and Setup

There isn't any special setup for it.

## Document Loader

See a [usage example](/docs/integrations/document_loaders/hacker_news).
"""
logger.info("# Hacker News")


logger.info("\n\n[DONE]", bright=True)