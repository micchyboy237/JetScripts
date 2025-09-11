from jet.logger import logger
from langchain_community.vectorstores import InfinispanVS
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
# Infinispan VS

> [Infinispan](https://infinispan.org) Infinispan is an open-source in-memory data grid that provides
> a key/value data store able to hold all types of data, from Java objects to plain text.
> Since version 15 Infinispan supports vector search over caches.

## Installation and Setup
See [Get Started](https://infinispan.org/get-started/) to run an Infinispan server, you may want to disable authentication
(not supported atm)

## Vector Store

See a [usage example](/docs/integrations/vectorstores/infinispanvs).
"""
logger.info("# Infinispan VS")


logger.info("\n\n[DONE]", bright=True)