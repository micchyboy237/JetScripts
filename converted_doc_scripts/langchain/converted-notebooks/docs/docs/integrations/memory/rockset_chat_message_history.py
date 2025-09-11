from jet.logger import logger
from langchain_community.chat_message_histories import (
RocksetChatMessageHistory,
)
from rockset import Regions, RocksetClient
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
# Rockset

>[Rockset](https://rockset.com/product/) is a real-time analytics database service for serving low latency, high concurrency analytical queries at scale. It builds a Converged Index™ on structured and semi-structured data with an efficient store for vector embeddings. Its support for running SQL on schemaless data makes it a perfect choice for running vector search with metadata filters. 


This notebook goes over how to use [Rockset](https://rockset.com/docs) to store chat message history.

## Setting up
"""
logger.info("# Rockset")

# %pip install --upgrade --quiet  rockset langchain-community

"""
To begin, with get your API key from the [Rockset console](https://console.rockset.com/apikeys). Find your API region for the Rockset [API reference](https://rockset.com/docs/rest-api#introduction).

## Example
"""
logger.info("## Example")


history = RocksetChatMessageHistory(
    session_id="MySession",
    client=RocksetClient(
        host=Regions.usw2a1,  # us-west-2 Oregon
    ),
    collection="langchain_demo",
    sync=True,
)
history.add_user_message("hi!")
history.add_ai_message("whats up?")
logger.debug(history.messages)

"""
The output should be something like:

```python
[
    HumanMessage(content='hi!', additional_kwargs={'id': '2e62f1c2-e9f7-465e-b551-49bae07fe9f0'}, example=False), 
    AIMessage(content='whats up?', additional_kwargs={'id': 'b9be8eda-4c18-4cf8-81c3-e91e876927d0'}, example=False)
]

```
"""
logger.info("The output should be something like:")

logger.info("\n\n[DONE]", bright=True)