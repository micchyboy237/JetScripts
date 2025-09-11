from jet.logger import logger
from langchain_community.vectorstores import PGEmbedding
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
# Postgres Embedding

> [pg_embedding](https://github.com/neondatabase/pg_embedding) is an open-source package for
> vector similarity search using `Postgres` and the `Hierarchical Navigable Small Worlds`
> algorithm for approximate nearest neighbor search.

## Installation and Setup

We need to install several python packages.
"""
logger.info("# Postgres Embedding")

pip install psycopg2-binary

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/pgembedding).
"""
logger.info("## Vector Store")


logger.info("\n\n[DONE]", bright=True)