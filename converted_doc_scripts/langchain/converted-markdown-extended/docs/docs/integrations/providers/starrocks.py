from jet.logger import logger
from langchain_community.vectorstores import StarRocks
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
# StarRocks

>[StarRocks](https://www.starrocks.io/) is a High-Performance Analytical Database.
`StarRocks` is a next-gen sub-second MPP database for full analytics scenarios, including multi-dimensional analytics, real-time analytics and ad-hoc query.

>Usually `StarRocks` is categorized into OLAP, and it has showed excellent performance in [ClickBench — a Benchmark For Analytical DBMS](https://benchmark.clickhouse.com/). Since it has a super-fast vectorized execution engine, it could also be used as a fast vectordb.

## Installation and Setup
"""
logger.info("# StarRocks")

pip install pymysql

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/starrocks).
"""
logger.info("## Vector Store")


logger.info("\n\n[DONE]", bright=True)