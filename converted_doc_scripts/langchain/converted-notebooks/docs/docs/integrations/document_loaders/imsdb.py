from jet.logger import logger
from langchain_community.document_loaders import IMSDbLoader
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
# IMSDb

>[IMSDb](https://imsdb.com/) is the `Internet Movie Script Database`.

This covers how to load `IMSDb` webpages into a document format that we can use downstream.
"""
logger.info("# IMSDb")


loader = IMSDbLoader("https://imsdb.com/scripts/BlacKkKlansman.html")

data = loader.load()

data[0].page_content[:500]

data[0].metadata

logger.info("\n\n[DONE]", bright=True)