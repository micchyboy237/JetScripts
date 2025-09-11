from jet.logger import logger
from langchain_community.document_loaders import DuckDBLoader
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
# DuckDB

>[DuckDB](https://duckdb.org/) is an in-process SQL OLAP database management system.

Load a `DuckDB` query with one document per row.
"""
logger.info("# DuckDB")

# %pip install --upgrade --quiet  duckdb


# %%file example.csv
Team,Payroll
Nationals,81.34
Reds,82.20

loader = DuckDBLoader("SELECT * FROM read_csv_auto('example.csv')")

data = loader.load()

logger.debug(data)

"""
## Specifying Which Columns are Content vs Metadata
"""
logger.info("## Specifying Which Columns are Content vs Metadata")

loader = DuckDBLoader(
    "SELECT * FROM read_csv_auto('example.csv')",
    page_content_columns=["Team"],
    metadata_columns=["Payroll"],
)

data = loader.load()

logger.debug(data)

"""
## Adding Source to Metadata
"""
logger.info("## Adding Source to Metadata")

loader = DuckDBLoader(
    "SELECT Team, Payroll, Team As source FROM read_csv_auto('example.csv')",
    metadata_columns=["source"],
)

data = loader.load()

logger.debug(data)

logger.info("\n\n[DONE]", bright=True)