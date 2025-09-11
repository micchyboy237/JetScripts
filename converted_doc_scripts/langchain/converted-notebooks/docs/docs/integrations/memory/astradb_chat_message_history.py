from jet.logger import logger
from langchain_astradb import AstraDBChatMessageHistory
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
# Astra DB 

> [DataStax Astra DB](https://docs.datastax.com/en/astra-db-serverless/index.html) is a serverless 
> AI-ready database built on `Apache Cassandra®` and made conveniently availablev
> through an easy-to-use JSON API.

This notebook goes over how to use Astra DB to store chat message history.

## Setup

To run this notebook you need a running Astra DB. Get the connection secrets on your Astra dashboard:

- the API Endpoint looks like `https://01234567-89ab-cdef-0123-456789abcdef-us-east1.apps.astra.datastax.com`;
- the Database Token looks like `AstraCS:aBcD0123...`.
"""
logger.info("# Astra DB")

# !pip install "langchain-astradb>=0.6,<0.7"

"""
### Set up the database connection parameters and secrets
"""
logger.info("### Set up the database connection parameters and secrets")

# import getpass

ASTRA_DB_API_ENDPOINT = input("ASTRA_DB_API_ENDPOINT = ")
# ASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")

"""
## Example
"""
logger.info("## Example")


message_history = AstraDBChatMessageHistory(
    session_id="test-session",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

message_history.add_user_message("hi!")

message_history.add_ai_message("hello, how are you?")

"""
[**API Reference:** `AstraDBChatMessageHistory`](https://python.langchain.com/api_reference/astradb/chat_message_histories/langchain_astradb.chat_message_histories.AstraDBChatMessageHistory.html#langchain_astradb.chat_message_histories.AstraDBChatMessageHistory)
"""

message_history.messages

logger.info("\n\n[DONE]", bright=True)