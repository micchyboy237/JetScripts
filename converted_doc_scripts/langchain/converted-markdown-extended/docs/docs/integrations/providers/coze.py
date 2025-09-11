from jet.logger import logger
from langchain_community.chat_models import ChatCoze
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
# Coze

[Coze](https://www.coze.com/) is an AI chatbot development platform that enables
the creation and deployment of chatbots for handling diverse conversations across
various applications.


## Installation and Setup

First, you need to get the `API_KEY` from the [Coze](https://www.coze.com/) website.


## Chat models

See a [usage example](/docs/integrations/chat/coze/).
"""
logger.info("# Coze")


logger.info("\n\n[DONE]", bright=True)