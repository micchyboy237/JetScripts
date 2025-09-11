from jet.logger import logger
from langchain_community.document_loaders import RedditPostsLoader
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
# Reddit

>[Reddit](https://www.reddit.com) is an American social news aggregation, content rating, and discussion website.

## Installation and Setup

First, you need to install a python package.
"""
logger.info("# Reddit")

pip install praw

"""
Make a [Reddit Application](https://www.reddit.com/prefs/apps/) and initialize the loader with your Reddit API credentials.

## Document Loader

See a [usage example](/docs/integrations/document_loaders/reddit).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)