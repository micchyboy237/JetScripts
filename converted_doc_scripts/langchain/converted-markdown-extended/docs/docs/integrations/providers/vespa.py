from jet.logger import logger
from langchain.retrievers import VespaRetriever
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
# Vespa

>[Vespa](https://vespa.ai/) is a fully featured search engine and vector database.
> It supports vector search (ANN), lexical search, and search in structured data, all in the same query.

## Installation and Setup
"""
logger.info("# Vespa")

pip install pyvespa

"""
## Retriever

See a [usage example](/docs/integrations/retrievers/vespa).
"""
logger.info("## Retriever")


logger.info("\n\n[DONE]", bright=True)