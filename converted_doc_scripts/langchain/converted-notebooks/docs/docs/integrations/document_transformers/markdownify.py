from jet.logger import logger
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
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
# Markdownify

> [markdownify](https://github.com/matthewwithanm/python-markdownify) is a Python package that converts HTML documents to Markdown format with customizable options for handling tags (links, images, ...), heading styles and other.
"""
logger.info("# Markdownify")

# %pip install --upgrade --quiet  markdownify


urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

docs


md = MarkdownifyTransformer()
converted_docs = md.transform_documents(docs)

logger.debug(converted_docs[0].page_content[:1000])

md = MarkdownifyTransformer(strip="a")
converted_docs = md.transform_documents(docs)

logger.debug(converted_docs[0].page_content[:1000])

md = MarkdownifyTransformer(strip=["h1", "a"])
converted_docs = md.transform_documents(docs)

logger.debug(converted_docs[0].page_content[:1000])

logger.info("\n\n[DONE]", bright=True)