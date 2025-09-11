from jet.logger import logger
from langchain_community.vectorstores import Annoy
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
# Annoy

> [Annoy](https://github.com/spotify/annoy) (`Approximate Nearest Neighbors Oh Yeah`)
> is a C++ library with Python bindings to search for points in space that are
> close to a given query point. It also creates large read-only file-based data
> structures that are mapped into memory so that many processes may share the same data.

## Installation and Setup
"""
logger.info("# Annoy")

pip install annoy

"""
## Vectorstore

See a [usage example](/docs/integrations/vectorstores/annoy).
"""
logger.info("## Vectorstore")


logger.info("\n\n[DONE]", bright=True)