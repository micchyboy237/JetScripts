from jet.logger import logger
from langchain_falkordb.message_history import (
FalkorDBChatMessageHistory,
)
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
# FalkorDB

<a href='https://docs.falkordb.com/' target='_blank'>FalkorDB</a> is an open-source graph database management system, renowned for its efficient management of highly connected data. Unlike traditional databases that store data in tables, FalkorDB uses a graph structure with nodes, edges, and properties to represent and store data. This design allows for high-performance queries on complex data relationships.

This notebook goes over how to use `FalkorDB` to store chat message history

**NOTE**: You can use FalkorDB locally or use FalkorDB Cloud. <a href='https://docs.falkordb.com/' target='blank'>See installation instructions</a>
"""
logger.info("# FalkorDB")

host = "localhost"
port = 6379


history = FalkorDBChatMessageHistory(host=host, port=port, session_id="session_id_1")

history.add_user_message("hi!")

history.add_ai_message("whats up?")

history.messages

logger.info("\n\n[DONE]", bright=True)