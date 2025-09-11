from jet.logger import logger
from langchain_airbyte import AirbyteLoader
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
# Airbyte

>[Airbyte](https://github.com/airbytehq/airbyte) is a data integration platform for ELT pipelines from APIs,
> databases & files to warehouses & lakes. It has the largest catalog of ELT connectors to data warehouses and databases.

## Installation and Setup
"""
logger.info("# Airbyte")

pip install -U langchain-airbyte

"""
:::note

Currently, the `langchain-airbyte` library does not support Pydantic v2.
Please downgrade to Pydantic v1 to use this package.

This package also currently requires Python 3.10+.

:::

The integration package doesn't require any global environment variables that need to be
set, but some integrations (e.g. `source-github`) may need credentials passed in.

## Document loader

### AirbyteLoader

See a [usage example](/docs/integrations/document_loaders/airbyte).
"""
logger.info("## Document loader")


logger.info("\n\n[DONE]", bright=True)