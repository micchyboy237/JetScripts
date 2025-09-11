from jet.logger import logger
from langchain_neo4j import Neo4jChatMessageHistory
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
# Neo4j

[Neo4j](https://en.wikipedia.org/wiki/Neo4j) is an open-source graph database management system, renowned for its efficient management of highly connected data. Unlike traditional databases that store data in tables, Neo4j uses a graph structure with nodes, edges, and properties to represent and store data. This design allows for high-performance queries on complex data relationships.

This notebook goes over how to use `Neo4j` to store chat message history.
"""
logger.info("# Neo4j")


history = Neo4jChatMessageHistory(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password",
    session_id="session_id_1",
)

history.add_user_message("hi!")

history.add_ai_message("whats up?")

history.messages

logger.info("\n\n[DONE]", bright=True)