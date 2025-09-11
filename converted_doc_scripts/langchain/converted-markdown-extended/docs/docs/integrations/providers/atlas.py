from jet.logger import logger
from langchain_community.vectorstores import AtlasDB
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
# Atlas

>[Nomic Atlas](https://docs.nomic.ai/index.html) is a platform for interacting with both
> small and internet scale unstructured datasets.


## Installation and Setup

- Install the Python package with `pip install nomic`
- `Nomic` is also included in langchains poetry extras `poetry install -E all`


## VectorStore

See a [usage example](/docs/integrations/vectorstores/atlas).
"""
logger.info("# Atlas")


logger.info("\n\n[DONE]", bright=True)