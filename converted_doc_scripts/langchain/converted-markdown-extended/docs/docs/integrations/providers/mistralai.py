from jet.logger import logger
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
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
# MistralAI

>[Mistral AI](https://docs.mistral.ai/api/) is a platform that offers hosting for their powerful open source models.


## Installation and Setup

A valid [API key](https://console.mistral.ai/users/api-keys/) is needed to communicate with the API.

You will also need the `langchain-mistralai` package:
"""
logger.info("# MistralAI")

pip install langchain-mistralai

"""
## Chat models

### ChatMistralAI

See a [usage example](/docs/integrations/chat/mistralai).
"""
logger.info("## Chat models")


"""
## Embedding models

### MistralAIEmbeddings

See a [usage example](/docs/integrations/text_embedding/mistralai).
"""
logger.info("## Embedding models")


logger.info("\n\n[DONE]", bright=True)