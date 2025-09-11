from jet.logger import logger
from langchain_community.document_loaders import DataFrameLoader
import os
import pandas as pd
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
# Pandas DataFrame

This notebook goes over how to load data from a [pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/index) DataFrame.
"""
logger.info("# Pandas DataFrame")

# %pip install --upgrade --quiet  pandas


df = pd.read_csv("example_data/mlb_teams_2012.csv")

df.head()


loader = DataFrameLoader(df, page_content_column="Team")

loader.load()

for i in loader.lazy_load():
    logger.debug(i)

logger.info("\n\n[DONE]", bright=True)