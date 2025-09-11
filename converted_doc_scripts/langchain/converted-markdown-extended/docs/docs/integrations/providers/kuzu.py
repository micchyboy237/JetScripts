from jet.logger import logger
from langchain_kuzu.chains.graph_qa.kuzu import KuzuQAChain
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph
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
# Kùzu

> [Kùzu](https://kuzudb.com/) is an embeddable, scalable, extremely fast graph database.
> It is permissively licensed with an MIT license, and you can see its source code [here](https://github.com/kuzudb/kuzu).

> Key characteristics of Kùzu:
>- Performance and scalability: Implements modern, state-of-the-art join algorithms for graphs.
>- Usability: Very easy to set up and get started with, as there are no servers (embedded architecture).
>- Interoperability: Can conveniently scan and copy data from external columnar formats, CSV, JSON and relational databases.
>- Structured property graph model: Implements the property graph model, with added structure.
>- Cypher support: Allows convenient querying of the graph in Cypher, a declarative query language.

> Get started with Kùzu by visiting their [documentation](https://docs.kuzudb.com/).


## Installation and Setup

Install the Python SDK as follows:
"""
logger.info("# Kùzu")

pip install -U langchain-kuzu

"""
## Usage

## Graphs

See a [usage example](/docs/integrations/graphs/kuzu_db).
"""
logger.info("## Usage")


"""
## Chains

See a [usage example](/docs/integrations/graphs/kuzu_db/#creating-kuzuqachain).
"""
logger.info("## Chains")


logger.info("\n\n[DONE]", bright=True)