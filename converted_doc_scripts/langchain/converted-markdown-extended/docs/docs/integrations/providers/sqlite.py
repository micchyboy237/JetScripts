from jet.logger import logger
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.vectorstores import SQLiteVSS # legacy
from langchain_community.vectorstores import SQLiteVec
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
# SQLite

>[SQLite](https://en.wikipedia.org/wiki/SQLite) is a database engine written in the
> C programming language. It is not a standalone app; rather, it is a library that
> software developers embed in their apps. As such, it belongs to the family of
> embedded databases. It is the most widely deployed database engine, as it is
> used by several of the top web browsers, operating systems, mobile phones, and other embedded systems.

## Installation and Setup

We need to install the `SQLAlchemy` python package.
"""
logger.info("# SQLite")

pip install SQLAlchemy

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/sqlitevec).
"""
logger.info("## Vector Store")


"""
## Memory

See a [usage example](/docs/integrations/memory/sqlite).
"""
logger.info("## Memory")


logger.info("\n\n[DONE]", bright=True)