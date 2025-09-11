from jet.logger import logger
from langchain_community.retrievers import DriaRetriever
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
# Dria

>[Dria](https://dria.co/) is a hub of public RAG models for developers to
> both contribute and utilize a shared embedding lake.

See more details about the LangChain integration with Dria
at [this page](https://dria.co/docs/integrations/langchain).

## Installation and Setup

You have to install a python package:
"""
logger.info("# Dria")

pip install dria

"""
You have to get an API key from Dria. You can get it by signing up at [Dria](https://dria.co/).

## Retrievers

See a [usage example](/docs/integrations/retrievers/dria_index).
"""
logger.info("## Retrievers")


logger.info("\n\n[DONE]", bright=True)