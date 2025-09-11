from jet.logger import logger
from langchain_community.vectorstores import Clickhouse, ClickhouseSettings
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
# ClickHouse

> [ClickHouse](https://clickhouse.com/) is the fast and resource efficient open-source database for real-time
> apps and analytics with full SQL support and a wide range of functions to assist users in writing analytical queries.
> It has data structures and distance search functions (like `L2Distance`) as well as
> [approximate nearest neighbor search indexes](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/annindexes)
> That enables ClickHouse to be used as a high performance and scalable vector database to store and search vectors with SQL.


## Installation and Setup

We need to install `clickhouse-connect` python package.
"""
logger.info("# ClickHouse")

pip install clickhouse-connect

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/clickhouse).
"""
logger.info("## Vector Store")


logger.info("\n\n[DONE]", bright=True)