from jet.logger import logger
from langchain_community.document_loaders import DuckDBLoader
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
# DuckDB

>[DuckDB](https://duckdb.org/) is an in-process SQL OLAP database management system.

## Installation and Setup

First, you need to install `duckdb` python package.
"""
logger.info("# DuckDB")

pip install duckdb

"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/duckdb).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)