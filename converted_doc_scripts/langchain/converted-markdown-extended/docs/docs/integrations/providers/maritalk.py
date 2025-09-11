from jet.logger import logger
from langchain_community.chat_models import ChatMaritalk
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
# MariTalk

>[MariTalk](https://www.maritaca.ai/en) is an LLM-based chatbot trained to meet the needs of Brazil.

## Installation and Setup

You have to get the MariTalk API key.

You also need to install the `httpx` Python package.
"""
logger.info("# MariTalk")

pip install httpx

"""
## Chat models

See a [usage example](/docs/integrations/chat/maritalk).
"""
logger.info("## Chat models")


logger.info("\n\n[DONE]", bright=True)