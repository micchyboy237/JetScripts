from jet.logger import logger
from langchain_community.retrievers import BM25Retriever
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
# rank_bm25

[rank_bm25](https://github.com/dorianbrown/rank_bm25) is an open-source collection of algorithms
designed to query documents and return the most relevant ones, commonly used for creating
search engines.

See its [project page](https://github.com/dorianbrown/rank_bm25) for available algorithms.


## Installation and Setup

First, you need to install `rank_bm25` python package.
"""
logger.info("# rank_bm25")

pip install rank_bm25

"""
## Retriever

See a [usage example](/docs/integrations/retrievers/bm25).
"""
logger.info("## Retriever")


logger.info("\n\n[DONE]", bright=True)