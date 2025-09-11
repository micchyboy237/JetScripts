from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaLLM
from jet.logger import logger
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
# Ollama

>[Ollama](https://www.anthropic.com/) is an AI safety and research company, and is the creator of `Claude`.
This page covers all integrations between `Ollama` models and `LangChain`.

## Installation and Setup

To use `Ollama` models, you need to install a python package:
"""
logger.info("# Ollama")

pip install -U langchain-anthropic

"""
# You need to set the `ANTHROPIC_API_KEY` environment variable.
You can get an Ollama API key [here](https://console.anthropic.com/settings/keys)

## Chat Models

### ChatOllama

See a [usage example](/docs/integrations/chat/anthropic).
"""
logger.info("## Chat Models")


model = ChatOllama(model="llama3.2")

"""
## LLMs

### [Legacy] OllamaLLM

**NOTE**: `OllamaLLM` only supports legacy `Claude 2` models.
To use the newest `Claude 3` models, please use `ChatOllama` instead.

See a [usage example](/docs/integrations/llms/anthropic).
"""
logger.info("## LLMs")


model = OllamaLLM(model='claude-2.1')

logger.info("\n\n[DONE]", bright=True)