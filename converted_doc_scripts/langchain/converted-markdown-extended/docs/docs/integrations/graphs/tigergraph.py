from jet.logger import logger
from langchain_community.graphs import TigerGraph
import os
import pyTigerGraph as tg
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

A big example of the `TigerGraph` and `LangChain` integration [presented here](https://github.com/tigergraph/graph-ml-notebooks/blob/main/applications/large_language_models/TigerGraph_LangChain_Demo.ipynb).

## Installation and Setup

Follow instructions [how to connect to the `TigerGraph` database](https://docs.tigergraph.com/pytigergraph/current/getting-started/connection).

Install the Python SDK:
"""
logger.info("# TigerGraph")

pip install pyTigerGraph

"""
## Example

To utilize the `TigerGraph InquiryAI` functionality, you can import `TigerGraph` from `langchain_community.graphs`.
"""
logger.info("## Example")


conn = tg.TigerGraphConnection(host="DATABASE_HOST_HERE", graphname="GRAPH_NAME_HERE", username="USERNAME_HERE", password="PASSWORD_HERE")

conn.ai.configureInquiryAIHost("INQUIRYAI_HOST_HERE")


graph = TigerGraph(conn)
result = graph.query("How many servers are there?")
logger.debug(result)

logger.info("\n\n[DONE]", bright=True)