from jet.logger import logger
from langchain_community.vectorstores import Tigris
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
# Tigris

> [Tigris](https://tigrisdata.com) is an open-source Serverless NoSQL Database and Search Platform designed to simplify building high-performance vector search applications.
> `Tigris` eliminates the infrastructure complexity of managing, operating, and synchronizing multiple tools, allowing you to focus on building great applications instead.

## Installation and Setup
"""
logger.info("# Tigris")

pip install tigrisdb openapi-schema-pydantic

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/tigris).
"""
logger.info("## Vector Store")


logger.info("\n\n[DONE]", bright=True)