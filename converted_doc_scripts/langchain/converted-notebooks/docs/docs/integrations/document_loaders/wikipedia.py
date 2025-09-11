from jet.logger import logger
from langchain_community.document_loaders import WikipediaLoader
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
# Wikipedia

>[Wikipedia](https://wikipedia.org/) is a multilingual free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing system called MediaWiki. `Wikipedia` is the largest and most-read reference work in history.

This notebook shows how to load wiki pages from `wikipedia.org` into the Document format that we use downstream.

## Installation

First, you need to install the `langchain_community` and `wikipedia` packages.
"""
logger.info("# Wikipedia")

# %pip install -qU langchain_community wikipedia

"""
## Parameters

`WikipediaLoader` has the following arguments:
- `query`: the free text which used to find documents in Wikipedia
- `lang` (optional): default="en". Use it to search in a specific language part of Wikipedia
- `load_max_docs` (optional): default=100. Use it to limit number of downloaded documents. It takes time to download all 100 documents, so use a small number for experiments. There is a hard limit of 300 for now.
- `load_all_available_meta` (optional): default=False. By default only the most important fields downloaded: `title` and `summary`. If `True` then all available fields will be downloaded.
- `doc_content_chars_max` (optional): default=4000. The maximum number of characters for the document content.

## Example
"""
logger.info("## Parameters")


docs = WikipediaLoader(query="HUNTER X HUNTER", load_max_docs=2).load()
len(docs)

docs[0].metadata  # metadata of the first document

docs[0].page_content[:400]  # a part of the page content

logger.info("\n\n[DONE]", bright=True)