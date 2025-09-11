from jet.logger import logger
from langchain_together import ChatTogether
from langchain_together import Together
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
# Together AI

[Together AI](https://www.together.ai/) offers an API to query [50+ leading open-source models](https://docs.together.ai/docs/inference-models) in a couple lines of code.

This example goes over how to use LangChain to interact with Together AI models.

## Installation
"""
logger.info("# Together AI")

# %pip install --upgrade langchain-together

"""
## Environment

To use Together AI, you'll need an API key which you can find here:
https://api.together.ai/settings/api-keys. This can be passed in as an init param
``together_api_key`` or set as environment variable ``TOGETHER_API_KEY``.

## Example
"""
logger.info("## Environment")


chat = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
)

for m in chat.stream("Tell me fun things to do in NYC"):
    logger.debug(m.content, end="", flush=True)


llm = Together(
    model="codellama/CodeLlama-70b-Python-hf",
)

logger.debug(llm.invoke("def bubble_sort(): "))

logger.info("\n\n[DONE]", bright=True)