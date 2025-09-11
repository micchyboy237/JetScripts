from jet.logger import logger
from langchain_community.vectorstores import Yellowbrick
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
# Yellowbrick

>[Yellowbrick](https://yellowbrick.com/) is a provider of
> Enterprise Data Warehousing, Ad-hoc and Streaming Analytics,
> BI and AI workloads.

## Vector store

We have to install a python package:
"""
logger.info("# Yellowbrick")

pip install psycopg2

"""

"""


logger.info("\n\n[DONE]", bright=True)