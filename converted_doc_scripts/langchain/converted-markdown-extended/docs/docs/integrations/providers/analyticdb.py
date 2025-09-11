from jet.logger import logger
from langchain_community.vectorstores import AnalyticDB
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
# AnalyticDB

>[AnalyticDB for PostgreSQL](https://www.alibabacloud.com/help/en/analyticdb-for-postgresql/latest/product-introduction-overview)
> is a massively parallel processing (MPP) data warehousing service
> from [Alibaba Cloud](https://www.alibabacloud.com/)
>that is designed to analyze large volumes of data online.

>`AnalyticDB for PostgreSQL` is developed based on the open-source `Greenplum Database`
> project and is enhanced with in-depth extensions by `Alibaba Cloud`. AnalyticDB
> for PostgreSQL is compatible with the ANSI SQL 2003 syntax and the PostgreSQL and
> Oracle database ecosystems. AnalyticDB for PostgreSQL also supports row store and
> column store. AnalyticDB for PostgreSQL processes petabytes of data offline at a
> high performance level and supports highly concurrent.

This page covers how to use the AnalyticDB ecosystem within LangChain.

## Installation and Setup

You need to install the `sqlalchemy` python package.
"""
logger.info("# AnalyticDB")

pip install sqlalchemy

"""
## VectorStore

See a [usage example](/docs/integrations/vectorstores/analyticdb).
"""
logger.info("## VectorStore")


logger.info("\n\n[DONE]", bright=True)