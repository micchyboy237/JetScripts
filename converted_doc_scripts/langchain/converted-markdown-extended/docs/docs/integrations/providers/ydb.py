from jet.logger import logger
from langchain_ydb.vectorstores import YDB
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
# YDB

All functionality related to YDB.

> [YDB](https://ydb.tech/) is a versatile open source Distributed SQL Database that combines
> high availability and scalability with strong consistency and ACID transactions.
> It accommodates transactional (OLTP), analytical (OLAP), and streaming workloads simultaneously.

## Installation and Setup
"""
logger.info("# YDB")

pip install langchain-ydb

"""
## Vector Store

To import YDB vector store:
"""
logger.info("## Vector Store")


"""
For a more detailed walkthrough of the YDB vector store, see [this notebook](/docs/integrations/vectorstores/ydb).
"""
logger.info("For a more detailed walkthrough of the YDB vector store, see [this notebook](/docs/integrations/vectorstores/ydb).")

logger.info("\n\n[DONE]", bright=True)