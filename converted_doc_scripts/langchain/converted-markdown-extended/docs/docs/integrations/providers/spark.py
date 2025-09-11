from jet.logger import logger
from langchain_community.agent_toolkits import SparkSQLToolkit, create_spark_sql_agent
from langchain_community.document_loaders import PySparkDataFrameLoader
from langchain_community.tools.spark_sql.tool import InfoSparkSQLTool
from langchain_community.tools.spark_sql.tool import ListSparkSQLTool
from langchain_community.tools.spark_sql.tool import QueryCheckerTool
from langchain_community.tools.spark_sql.tool import QuerySparkSQLTool
from langchain_community.utilities.spark_sql import SparkSQL
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
# Spark

>[Apache Spark](https://spark.apache.org/) is a unified analytics engine for
> large-scale data processing. It provides high-level APIs in Scala, Java,
> Python, and R, and an optimized engine that supports general computation
> graphs for data analysis. It also supports a rich set of higher-level
> tools including `Spark SQL` for SQL and DataFrames, `pandas API on Spark`
> for pandas workloads, `MLlib` for machine learning,
> `GraphX` for graph processing, and `Structured Streaming` for stream processing.

## Document loaders

### PySpark

It loads data from a `PySpark` DataFrame.

See a [usage example](/docs/integrations/document_loaders/pyspark_dataframe).
"""
logger.info("# Spark")


"""
## Tools/Toolkits

### Spark SQL toolkit

Toolkit for interacting with `Spark SQL`.

See a [usage example](/docs/integrations/tools/spark_sql).
"""
logger.info("## Tools/Toolkits")


"""
#### Spark SQL individual tools

You can use individual tools from the Spark SQL Toolkit:
- `InfoSparkSQLTool`: tool for getting metadata about a Spark SQL
- `ListSparkSQLTool`: tool for getting tables names
- `QueryCheckerTool`: tool uses an LLM to check if a query is correct
- `QuerySparkSQLTool`: tool for querying a Spark SQL
"""
logger.info("#### Spark SQL individual tools")


logger.info("\n\n[DONE]", bright=True)