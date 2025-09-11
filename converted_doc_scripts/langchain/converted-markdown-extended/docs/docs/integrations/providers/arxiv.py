from jet.logger import logger
from langchain_community.document_loaders import ArxivLoader
from langchain_community.retrievers import ArxivRetriever
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
# Arxiv

>[arXiv](https://arxiv.org/) is an open-access archive for 2 million scholarly articles in the fields of physics,
> mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and
> systems science, and economics.


## Installation and Setup

First, you need to install `arxiv` python package.
"""
logger.info("# Arxiv")

pip install arxiv

"""
Second, you need to install `PyMuPDF` python package which transforms PDF files downloaded from the `arxiv.org` site into the text format.
"""
logger.info("Second, you need to install `PyMuPDF` python package which transforms PDF files downloaded from the `arxiv.org` site into the text format.")

pip install pymupdf

"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/arxiv).
"""
logger.info("## Document Loader")


"""
## Retriever

See a [usage example](/docs/integrations/retrievers/arxiv).
"""
logger.info("## Retriever")


logger.info("\n\n[DONE]", bright=True)