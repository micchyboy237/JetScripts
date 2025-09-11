from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.agent_toolkits import SparkSQLToolkit, create_spark_sql_agent
from langchain_community.utilities.spark_sql import SparkSQL
from pyspark.sql import SparkSession
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
# Spark SQL Toolkit

This notebook shows how to use agents to interact with `Spark SQL`. Similar to [SQL Database Agent](/docs/integrations/tools/sql_database), it is designed to address general inquiries about `Spark SQL` and facilitate error recovery.

**NOTE: Note that, as this agent is in active development, all answers might not be correct. Additionally, it is not guaranteed that the agent won't perform DML statements on your Spark cluster given certain questions. Be careful running it on sensitive data!**

## Initialization
"""
logger.info("# Spark SQL Toolkit")



spark = SparkSession.builder.getOrCreate()
schema = "langchain_example"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema}")
spark.sql(f"USE {schema}")
csv_file_path = "titanic.csv"
table = "titanic"
spark.read.csv(csv_file_path, header=True, inferSchema=True).write.saveAsTable(table)
spark.table(table).show()

spark_sql = SparkSQL(schema=schema)
llm = ChatOllama(model="llama3.2")
toolkit = SparkSQLToolkit(db=spark_sql, llm=llm)
agent_executor = create_spark_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

"""
## Example: describing a table
"""
logger.info("## Example: describing a table")

agent_executor.run("Describe the titanic table")

"""
## Example: running queries
"""
logger.info("## Example: running queries")

agent_executor.run("whats the square root of the average age?")

agent_executor.run("What's the name of the oldest survived passenger?")

logger.info("\n\n[DONE]", bright=True)