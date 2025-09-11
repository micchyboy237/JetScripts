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

This loader fetches the text from the Tweets of a list of `Twitter` users, using the `tweepy` Python package.
You must initialize the loader with your `Twitter API` token, and you need to pass in the Twitter username you want to extract.
"""
logger.info("# Twitter")


# %pip install --upgrade --quiet  tweepy

loader = TwitterTweetLoader.from_bearer_token(
    oauth2_bearer_token="YOUR BEARER TOKEN",
    twitter_users=["elonmusk"],
    number_tweets=50,  # Default value is 100
)

documents = loader.load()
documents[:5]

logger.info("\n\n[DONE]", bright=True)