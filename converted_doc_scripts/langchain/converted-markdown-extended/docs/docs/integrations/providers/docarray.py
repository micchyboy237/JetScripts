from jet.logger import logger
from langchain_community.retrievers import DocArrayRetriever
from langchain_community.vectorstores import DocArrayHnswSearch
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
# DocArray

> [DocArray](https://docarray.jina.ai/) is a library for nested, unstructured, multimodal data in transit,
> including text, image, audio, video, 3D mesh, etc. It allows deep-learning engineers to efficiently process,
> embed, search, recommend, store, and transfer multimodal data with a Pythonic API.


## Installation and Setup

We need to install `docarray` python package.
"""
logger.info("# DocArray")

pip install docarray

"""
## Vector Store

LangChain provides an access to the `In-memory` and `HNSW` vector stores from the `DocArray` library.

See a [usage example](/docs/integrations/vectorstores/docarray_hnsw).
"""
logger.info("## Vector Store")


"""
See a [usage example](/docs/integrations/vectorstores/docarray_in_memory).
"""
logger.info("See a [usage example](/docs/integrations/vectorstores/docarray_in_memory).")

from langchain_community.vectorstores DocArrayInMemorySearch

"""
## Retriever

See a [usage example](/docs/integrations/retrievers/docarray_retriever).
"""
logger.info("## Retriever")


logger.info("\n\n[DONE]", bright=True)