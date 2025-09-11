from jet.logger import logger
from langchain_community.document_loaders import TwitterTweetLoader
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
# Twitter

>[Twitter](https://twitter.com/) is an online social media and social networking service.


## Installation and Setup
"""
logger.info("# Twitter")

pip install tweepy

"""
We must initialize the loader with the `Twitter API` token, and we need to set up the Twitter `username`.


## Document Loader

See a [usage example](/docs/integrations/document_loaders/twitter).
"""
logger.info("## Document Loader")


"""
## Chat loader

See a [usage example](/docs/integrations/chat_loaders/twitter).
"""
logger.info("## Chat loader")

logger.info("\n\n[DONE]", bright=True)