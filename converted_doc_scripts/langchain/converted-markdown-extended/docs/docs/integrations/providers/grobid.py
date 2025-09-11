from jet.logger import logger
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser
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
# Grobid

GROBID is a machine learning library for extracting, parsing, and re-structuring raw documents.

It is designed and expected to be used to parse academic papers, where it works particularly well.

*Note*: if the articles supplied to Grobid are large documents (e.g. dissertations) exceeding a certain number
of elements, they might not be processed.

This page covers how to use the Grobid to parse articles for LangChain.

## Installation
The grobid installation is described in details in https://grobid.readthedocs.io/en/latest/Install-Grobid/.
However, it is probably easier and less troublesome to run grobid through a docker container,
as documented [here](https://grobid.readthedocs.io/en/latest/Grobid-docker/).

## Use Grobid with LangChain

Once grobid is installed and up and running (you can check by accessing it http://localhost:8070),
you're ready to go.

You can now use the GrobidParser to produce documents
"""
logger.info("# Grobid")


loader = GenericLoader.from_filesystem(
    "/Users/31treehaus/Desktop/Papers/",
    glob="*",
    suffixes=[".pdf"],
    parser= GrobidParser(segment_sentences=False)
)
docs = loader.load()

loader = GenericLoader.from_filesystem(
    "/Users/31treehaus/Desktop/Papers/",
    glob="*",
    suffixes=[".pdf"],
    parser= GrobidParser(segment_sentences=True)
)
docs = loader.load()

"""
Chunk metadata will include Bounding Boxes. Although these are a bit funky to parse,
they are explained in https://grobid.readthedocs.io/en/latest/Coordinates-in-PDF/
"""
logger.info("Chunk metadata will include Bounding Boxes. Although these are a bit funky to parse,")

logger.info("\n\n[DONE]", bright=True)