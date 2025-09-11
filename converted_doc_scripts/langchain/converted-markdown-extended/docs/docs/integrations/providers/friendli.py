from jet.logger import logger
from langchain_community.chat_models.friendli import ChatFriendli
from langchain_community.llms.friendli import Friendli
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
# Friendli AI

> [FriendliAI](https://friendli.ai/) enhances AI application performance and optimizes
> cost savings with scalable, efficient deployment options, tailored for high-demand AI workloads.

## Installation and setup

Install the `friendli-client` python package.
"""
logger.info("# Friendli AI")

pip install -U langchain_community friendli-client

"""
Sign in to [Friendli Suite](https://suite.friendli.ai/) to create a Personal Access Token,
and set it as the `FRIENDLI_TOKEN` environment variable.


## Chat models

See a [usage example](/docs/integrations/chat/friendli).
"""
logger.info("## Chat models")


chat = ChatFriendli(model='meta-llama-3.1-8b-instruct')

for m in chat.stream("Tell me fun things to do in NYC"):
    logger.debug(m.content, end="", flush=True)

"""
## LLMs

See a [usage example](/docs/integrations/llms/friendli).
"""
logger.info("## LLMs")


llm = Friendli(model='meta-llama-3.1-8b-instruct')

logger.debug(llm.invoke("def bubble_sort(): "))

logger.info("\n\n[DONE]", bright=True)