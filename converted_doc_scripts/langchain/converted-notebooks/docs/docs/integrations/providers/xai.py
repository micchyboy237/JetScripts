from jet.logger import logger
from langchain_xai import ChatXAI
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
# xAI

[xAI](https://console.x.ai) offers an API to interact with Grok models.

This example goes over how to use LangChain to interact with xAI models.

## Installation
"""
logger.info("# xAI")

# %pip install --upgrade langchain-xai

"""
## Environment

To use xAI, you'll need to [create an API key](https://console.x.ai/). The API key can be passed in as an init param ``xai_api_key`` or set as environment variable ``XAI_API_KEY``.

## Example

See [ChatXAI docs](/docs/integrations/chat/xai) for detail and supported features.
"""
logger.info("## Environment")


chat = ChatXAI(
    model="grok-4",
)

for m in chat.stream("Tell me fun things to do in NYC"):
    logger.debug(m.content, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)