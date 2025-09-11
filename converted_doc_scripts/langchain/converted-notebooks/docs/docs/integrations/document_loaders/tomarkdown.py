from jet.logger import logger
from langchain_community.document_loaders import ToMarkdownLoader
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
# 2Markdown

>[2markdown](https://2markdown.com/) service transforms website content into structured markdown files.
"""
logger.info("# 2Markdown")




loader = ToMarkdownLoader(url="/docs/get_started/introduction", api_key=api_key)

docs = loader.load()

logger.debug(docs[0].page_content)

logger.info("\n\n[DONE]", bright=True)