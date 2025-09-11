from jet.logger import logger
from langchain_community.vectorstores import Bagel
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
# BagelDB

> [BagelDB](https://www.bageldb.ai/) (`Open Vector Database for AI`), is like GitHub for AI data.
It is a collaborative platform where users can create,
share, and manage vector datasets. It can support private projects for independent developers,
internal collaborations for enterprises, and public contributions for data DAOs.

## Installation and Setup
"""
logger.info("# BagelDB")

pip install betabageldb

"""
## VectorStore

See a [usage example](/docs/integrations/vectorstores/bageldb).
"""
logger.info("## VectorStore")


logger.info("\n\n[DONE]", bright=True)