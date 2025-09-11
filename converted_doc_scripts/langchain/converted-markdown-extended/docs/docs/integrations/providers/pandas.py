from jet.logger import logger
from langchain_community.document_loaders import DataFrameLoader
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
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
# Pandas

>[pandas](https://pandas.pydata.org) is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
built on top of the `Python` programming language.

## Installation and Setup

Install the `pandas` package using `pip`:
"""
logger.info("# Pandas")

pip install pandas

"""
## Document loader

See a [usage example](/docs/integrations/document_loaders/pandas_dataframe).
"""
logger.info("## Document loader")


"""
## Toolkit

See a [usage example](/docs/integrations/tools/pandas).
"""
logger.info("## Toolkit")


logger.info("\n\n[DONE]", bright=True)