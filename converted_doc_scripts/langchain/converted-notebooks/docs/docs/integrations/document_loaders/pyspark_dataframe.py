from jet.logger import logger
from langchain_community.document_loaders import PySparkDataFrameLoader
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
# PySpark

This notebook goes over how to load data from a [PySpark](https://spark.apache.org/docs/latest/api/python/) DataFrame.
"""
logger.info("# PySpark")

# %pip install --upgrade --quiet  pyspark


spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("example_data/mlb_teams_2012.csv", header=True)


loader = PySparkDataFrameLoader(spark, df, page_content_column="Team")

loader.load()

logger.info("\n\n[DONE]", bright=True)