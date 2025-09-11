from jet.logger import logger
from langchain_community.document_loaders import RSSFeedLoader
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
# RSS Feeds

This covers how to load HTML news articles from a list of RSS feed URLs into a document format that we can use downstream.
"""
logger.info("# RSS Feeds")

# %pip install --upgrade --quiet  feedparser newspaper3k listparser


urls = ["https://news.ycombinator.com/rss"]

"""
Pass in urls to load them into Documents
"""
logger.info("Pass in urls to load them into Documents")

loader = RSSFeedLoader(urls=urls)
data = loader.load()
logger.debug(len(data))

logger.debug(data[0].page_content)

"""
You can pass arguments to the NewsURLLoader which it uses to load articles.
"""
logger.info("You can pass arguments to the NewsURLLoader which it uses to load articles.")

loader = RSSFeedLoader(urls=urls, nlp=True)
data = loader.load()
logger.debug(len(data))

data[0].metadata["keywords"]

data[0].metadata["summary"]

"""
You can also use an OPML file such as a Feedly export.  Pass in either a URL or the OPML contents.
"""
logger.info("You can also use an OPML file such as a Feedly export.  Pass in either a URL or the OPML contents.")

with open("example_data/sample_rss_feeds.opml", "r") as f:
    loader = RSSFeedLoader(opml=f.read())
data = loader.load()
logger.debug(len(data))

data[0].page_content

logger.info("\n\n[DONE]", bright=True)