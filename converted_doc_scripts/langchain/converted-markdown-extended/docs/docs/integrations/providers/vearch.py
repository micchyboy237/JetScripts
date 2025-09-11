from jet.logger import logger
from langchain_community.vectorstores import Vearch
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
# Vearch

[Vearch](https://github.com/vearch/vearch) is a scalable distributed system for efficient similarity search of deep learning vectors.

# Installation and Setup

Vearch Python SDK enables vearch to use locally. Vearch python sdk can be installed easily by pip install vearch.

# Vectorstore

Vearch also can used as vectorstore. Most details in [this notebook](/docs/integrations/vectorstores/vearch)
"""
logger.info("# Vearch")


logger.info("\n\n[DONE]", bright=True)