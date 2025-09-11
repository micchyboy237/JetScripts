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

>[Hacker News](https://en.wikipedia.org/wiki/Hacker_News) (sometimes abbreviated as `HN`) is a social news website focusing on computer science and entrepreneurship. It is run by the investment fund and startup incubator `Y Combinator`. In general, content that can be submitted is defined as "anything that gratifies one's intellectual curiosity."

This notebook covers how to pull page data and comments from [Hacker News](https://news.ycombinator.com/)
"""
logger.info("# Hacker News")


loader = HNLoader("https://news.ycombinator.com/item?id=34817881")

data = loader.load()

data[0].page_content[:300]

data[0].metadata

logger.info("\n\n[DONE]", bright=True)