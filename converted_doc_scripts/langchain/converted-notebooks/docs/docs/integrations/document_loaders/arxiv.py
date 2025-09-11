from jet.logger import logger
from langchain_community.document_loaders import ArxivLoader
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
# ArxivLoader

[arXiv](https://arxiv.org/) is an open-access archive for 2 million scholarly articles in the fields of physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems science, and economics.

## Setup

To access Arxiv document loader you'll need to install the `arxiv`, `PyMuPDF` and `langchain-community` integration packages. PyMuPDF transforms PDF files downloaded from the arxiv.org site into the text format.
"""
logger.info("# ArxivLoader")

# %pip install -qU langchain-community arxiv pymupdf

"""
## Instantiation

Now we can instantiate our model object and load documents:
"""
logger.info("## Instantiation")


loader = ArxivLoader(
    query="reasoning",
    load_max_docs=2,
)

"""
## Load

Use ``.load()`` to synchronously load into memory all Documents, with one
Document per one arxiv paper.

Let's run through a basic example of how to use the `ArxivLoader` searching for papers of reasoning:
"""
logger.info("## Load")

docs = loader.load()
docs[0]

logger.debug(docs[0].metadata)

"""
## Lazy Load

If we're loading a  large number of Documents and our downstream operations can be done over subsets of all loaded Documents, we can lazily load our Documents one at a time to minimize our memory footprint:
"""
logger.info("## Lazy Load")

docs = []

for doc in loader.lazy_load():
    docs.append(doc)

    if len(docs) >= 10:

        docs = []

"""
In this example we never have more than 10 Documents loaded into memory at a time.

## Use papers summaries as documents

You can use summaries of Arvix paper as documents rather than raw papers:
"""
logger.info("## Use papers summaries as documents")

docs = loader.get_summaries_as_docs()
docs[0]

"""
## API reference

For detailed documentation of all ArxivLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.arxiv.ArxivLoader.html#langchain_community.document_loaders.arxiv.ArxivLoader
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)