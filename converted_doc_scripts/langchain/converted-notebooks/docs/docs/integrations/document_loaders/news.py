from jet.logger import logger
from langchain_community.document_loaders import NewsURLLoader
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
# News URL

This covers how to load HTML news articles from a list of URLs into a document format that we can use downstream.
"""
logger.info("# News URL")


urls = [
    "https://www.bbc.com/news/world-us-canada-66388172",
    "https://www.bbc.com/news/entertainment-arts-66384971",
]

"""
Pass in urls to load them into Documents
"""
logger.info("Pass in urls to load them into Documents")

loader = NewsURLLoader(urls=urls)
data = loader.load()
logger.debug("First article: ", data[0])
logger.debug("\nSecond article: ", data[1])

"""
Use nlp=True to run nlp analysis and generate keywords + summary
"""
logger.info("Use nlp=True to run nlp analysis and generate keywords + summary")

loader = NewsURLLoader(urls=urls, nlp=True)
data = loader.load()
logger.debug("First article: ", data[0])
logger.debug("\nSecond article: ", data[1])

data[0].metadata["keywords"]

data[0].metadata["summary"]

logger.info("\n\n[DONE]", bright=True)