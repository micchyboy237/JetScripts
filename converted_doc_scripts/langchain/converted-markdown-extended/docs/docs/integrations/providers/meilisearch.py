from jet.logger import logger
from langchain_community.vectorstores import Meilisearch
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
# Meilisearch

> [Meilisearch](https://meilisearch.com) is an open-source, lightning-fast, and hyper
> relevant search engine.
> It comes with great defaults to help developers build snappy search experiences.
>
> You can [self-host Meilisearch](https://www.meilisearch.com/docs/learn/getting_started/installation#local-installation)
> or run on [Meilisearch Cloud](https://www.meilisearch.com/pricing).
>
>`Meilisearch v1.3` supports vector search.

## Installation and Setup

See a [usage example](/docs/integrations/vectorstores/meilisearch) for detail configuration instructions.


We need to install `meilisearch` python package.
"""
logger.info("# Meilisearch")

pip install meilisearch

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/meilisearch).
"""
logger.info("## Vector Store")


logger.info("\n\n[DONE]", bright=True)