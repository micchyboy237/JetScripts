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


This loader fetches the text from the Posts of Subreddits or Reddit users, using the `praw` Python package.

Make a [Reddit Application](https://www.reddit.com/prefs/apps/) and initialize the loader with your Reddit API credentials.
"""
logger.info("# Reddit")


# %pip install --upgrade --quiet  praw

loader = RedditPostsLoader(
    client_id="YOUR CLIENT ID",
    client_secret="YOUR CLIENT SECRET",
    user_agent="extractor by u/Master_Ocelot8179",
    categories=["new", "hot"],  # List of categories to load posts from
    mode="subreddit",
    search_queries=[
        "investing",
        "wallstreetbets",
    ],  # List of subreddits to load posts from
    number_posts=20,  # Default value is 10
)

documents = loader.load()
documents[:5]

logger.info("\n\n[DONE]", bright=True)