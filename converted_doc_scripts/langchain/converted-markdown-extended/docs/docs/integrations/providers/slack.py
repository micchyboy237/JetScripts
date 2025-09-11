from jet.logger import logger
from langchain_community.agent_toolkits import SlackToolkit
from langchain_community.chat_loaders.slack import SlackChatLoader
from langchain_community.document_loaders import SlackDirectoryLoader
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
# Slack

>[Slack](https://slack.com/) is an instant messaging program.

## Installation and Setup

There isn't any special setup for it.


## Document loader

See a [usage example](/docs/integrations/document_loaders/slack).
"""
logger.info("# Slack")


"""
## Toolkit

See a [usage example](/docs/integrations/tools/slack).
"""
logger.info("## Toolkit")


"""
## Chat loader

See a [usage example](/docs/integrations/chat_loaders/slack).
"""
logger.info("## Chat loader")


logger.info("\n\n[DONE]", bright=True)