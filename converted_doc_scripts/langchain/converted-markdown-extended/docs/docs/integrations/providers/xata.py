from jet.logger import logger
from langchain_community.chat_message_histories import XataChatMessageHistory
from langchain_community.vectorstores import XataVectorStore
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
# Xata

> [Xata](https://xata.io) is a serverless data platform, based on `PostgreSQL`.
> It provides a Python SDK for interacting with your database, and a UI
> for managing your data.
> `Xata` has a native vector type, which can be added to any table, and
> supports similarity search. LangChain inserts vectors directly to `Xata`,
> and queries it for the nearest neighbors of a given vector, so that you can
> use all the LangChain Embeddings integrations with `Xata`.


## Installation and Setup


We need to install `xata` python package.
"""
logger.info("# Xata")

pip install xata==1.0.0a7

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/xata).
"""
logger.info("## Vector Store")


"""
## Memory

See a [usage example](/docs/integrations/memory/xata_chat_message_history).
"""
logger.info("## Memory")


logger.info("\n\n[DONE]", bright=True)