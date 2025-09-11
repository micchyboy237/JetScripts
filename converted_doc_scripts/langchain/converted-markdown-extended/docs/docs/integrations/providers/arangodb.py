from arango import ArangoClient
from jet.logger import logger
from langchain.chains import ArangoGraphQAChain
from langchain_community.graphs import ArangoGraph
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
# ArangoDB

>[ArangoDB](https://github.com/arangodb/arangodb) is a scalable graph database system to
> drive value from connected data, faster. Native graphs, an integrated search engine, and JSON support, via a single query language. ArangoDB runs on-prem, in the cloud – anywhere.

## Installation and Setup

Install the [ArangoDB Python Driver](https://github.com/ArangoDB-Community/python-arango) package with
"""
logger.info("# ArangoDB")

pip install python-arango

"""
## Graph QA Chain

Connect your `ArangoDB` Database with a chat model to get insights on your data.

See the notebook example [here](/docs/integrations/graphs/arangodb).
"""
logger.info("## Graph QA Chain")



logger.info("\n\n[DONE]", bright=True)