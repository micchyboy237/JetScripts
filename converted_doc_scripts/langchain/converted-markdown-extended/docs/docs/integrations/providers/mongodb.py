from jet.logger import logger
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
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
# MongoDB

>[MongoDB](https://www.mongodb.com/) is a NoSQL, document-oriented
> database that supports JSON-like documents with a dynamic schema.

**NOTE:**
- See other `MongoDB` integrations on the [MongoDB Atlas page](/docs/integrations/providers/mongodb_atlas).

## Installation and Setup

Install the Python package:
"""
logger.info("# MongoDB")

pip install langchain-mongodb

"""
## Message Histories

See a [usage example](/docs/integrations/memory/mongodb_chat_message_history).

To import this vectorstore:
"""
logger.info("## Message Histories")


logger.info("\n\n[DONE]", bright=True)