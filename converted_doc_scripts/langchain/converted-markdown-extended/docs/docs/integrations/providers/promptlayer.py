from jet.logger import logger
from langchain.callbacks import PromptLayerCallbackHandler
from langchain_community.chat_models import PromptLayerChatOllama
from langchain_community.llms import PromptLayerOllama
import os
import promptlayer  # Don't forget this import!
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
# PromptLayer

>[PromptLayer](https://docs.promptlayer.com/introduction) is a platform for prompt engineering.
> It also helps with the LLM observability to visualize requests, version prompts, and track usage.
>
>While `PromptLayer` does have LLMs that integrate directly with LangChain (e.g.
> [`PromptLayerOllama`](https://docs.promptlayer.com/languages/langchain)),
> using a callback is the recommended way to integrate `PromptLayer` with LangChain.

## Installation and Setup

To work with `PromptLayer`, we have to:
- Create a `PromptLayer` account
- Create an api token and set it as an environment variable (`PROMPTLAYER_API_KEY`)

Install a Python package:
"""
logger.info("# PromptLayer")

pip install promptlayer

"""
## Callback

See a [usage example](/docs/integrations/callbacks/promptlayer).
"""
logger.info("## Callback")


"""
## LLM

See a [usage example](/docs/integrations/llms/promptlayer_ollama).
"""
logger.info("## LLM")


"""
## Chat Models

See a [usage example](/docs/integrations/chat/promptlayer_chatollama).
"""
logger.info("## Chat Models")


logger.info("\n\n[DONE]", bright=True)