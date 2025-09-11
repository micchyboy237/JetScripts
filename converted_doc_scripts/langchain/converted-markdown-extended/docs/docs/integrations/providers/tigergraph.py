from jet.logger import logger
from langchain_community.graphs import TigerGraph
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
# TigerGraph

>[TigerGraph](https://www.tigergraph.com/tigergraph-db/) is a natively distributed and high-performance graph database.
> The storage of data in a graph format of vertices and edges leads to rich relationships,
> ideal for grouding LLM responses.

## Installation and Setup

Follow instructions [how to connect to the `TigerGraph` database](https://docs.tigergraph.com/pytigergraph/current/getting-started/connection).

Install the Python SDK:
"""
logger.info("# TigerGraph")

pip install pyTigerGraph

"""
## Graph store

### TigerGraph

See a [usage example](/docs/integrations/graphs/tigergraph).
"""
logger.info("## Graph store")


logger.info("\n\n[DONE]", bright=True)