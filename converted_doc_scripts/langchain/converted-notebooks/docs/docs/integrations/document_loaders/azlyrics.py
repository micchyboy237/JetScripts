from jet.logger import logger
from langchain_community.document_loaders import AZLyricsLoader
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
# AZLyrics

>[AZLyrics](https://www.azlyrics.com/) is a large, legal, every day growing collection of lyrics.

This covers how to load AZLyrics webpages into a document format that we can use downstream.
"""
logger.info("# AZLyrics")


loader = AZLyricsLoader("https://www.azlyrics.com/lyrics/mileycyrus/flowers.html")

data = loader.load()

data

logger.info("\n\n[DONE]", bright=True)