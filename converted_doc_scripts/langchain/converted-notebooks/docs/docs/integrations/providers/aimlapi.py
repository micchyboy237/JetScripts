from jet.logger import logger
from langchain_aimlapi import AimlapiLLM
from langchain_aimlapi import ChatAimlapi
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
# AI/ML API LLM

[AI/ML API](https://aimlapi.com/app/?utm_source=langchain&utm_medium=github&utm_campaign=integration) provides an API to query **300+ leading AI models** (Deepseek, Gemini, ChatGPT, etc.) with enterprise-grade performance.

This example demonstrates how to use LangChain to interact with AI/ML API models.

## Installation
"""
logger.info("# AI/ML API LLM")

# %pip install --upgrade langchain-aimlapi

"""
## Environment

To use AI/ML API, you'll need an API key which you can generate at:
[https://aimlapi.com/app/](https://aimlapi.com/app/?utm_source=langchain&utm_medium=github&utm_campaign=integration)

You can pass it via `aimlapi_api_key` parameter or set as environment variable `AIMLAPI_API_KEY`.
"""
logger.info("## Environment")

# import getpass

if "AIMLAPI_API_KEY" not in os.environ:
#     os.environ["AIMLAPI_API_KEY"] = getpass.getpass("Enter your AI/ML API key: ")

"""
## Example: Chat Model
"""
logger.info("## Example: Chat Model")


chat = ChatAimlapi(
    model="meta-llama/Llama-3-70b-chat-hf",
)

for chunk in chat.stream("Tell me fun things to do in NYC"):
    logger.debug(chunk.content, end="", flush=True)

"""
## Example: Text Completion Model
"""
logger.info("## Example: Text Completion Model")


llm = AimlapiLLM(
    model="llama3.2",
)

logger.debug(llm.invoke("def bubble_sort(): "))

logger.info("\n\n[DONE]", bright=True)