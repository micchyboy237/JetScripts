from jet.adapters.langchain.chat_ollama import AzureChatOllama
from jet.adapters.langchain.chat_ollama import AzureOllama
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import Ollama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.adapters import ollama as lc_ollama
from langchain.chains import OllamaModerationChain
from langchain.retrievers import ChatGPTPluginRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.chatgpt import ChatGPTLoader
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
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
---
keywords: [ollama]
---

# Ollama

All functionality related to Ollama

> [Ollama](https://en.wikipedia.org/wiki/Ollama) is American artificial intelligence (AI) research laboratory
> consisting of the non-profit **Ollama Incorporated**
> and its for-profit subsidiary corporation **Ollama Limited Partnership**.
> **Ollama** conducts AI research with the declared intention of promoting and developing a friendly AI.
> **Ollama** systems run on an **Azure**-based supercomputing platform from **Microsoft**.
>
> The [Ollama API](https://platform.ollama.com/docs/models) is powered by a diverse set of models with different capabilities and price points.
>
> [ChatGPT](https://chat.ollama.com) is the Artificial Intelligence (AI) chatbot developed by `Ollama`.

## Installation and Setup

Install the integration package with
"""
logger.info("# Ollama")

pip install langchain-ollama

"""
# Get an Ollama api key and set it as an environment variable (`OPENAI_API_KEY`)

## Chat model

See a [usage example](/docs/integrations/chat/ollama).
"""
logger.info("## Chat model")


"""
If you are using a model hosted on `Azure`, you should use different wrapper for that:
"""
logger.info("If you are using a model hosted on `Azure`, you should use different wrapper for that:")


"""
For a more detailed walkthrough of the `Azure` wrapper, see [here](/docs/integrations/chat/azure_chat_ollama).

## LLM

See a [usage example](/docs/integrations/llms/ollama).
"""
logger.info("## LLM")


"""
If you are using a model hosted on `Azure`, you should use different wrapper for that:
"""
logger.info("If you are using a model hosted on `Azure`, you should use different wrapper for that:")


"""
For a more detailed walkthrough of the `Azure` wrapper, see [here](/docs/integrations/llms/azure_ollama).

## Embedding Model

See a [usage example](/docs/integrations/text_embedding/ollama)
"""
logger.info("## Embedding Model")


"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/chatgpt_loader).
"""
logger.info("## Document Loader")


"""
## Retriever

See a [usage example](/docs/integrations/retrievers/chatgpt-plugin).
"""
logger.info("## Retriever")


"""
## Tools

### Dall-E Image Generator

>[Ollama Dall-E](https://ollama.com/dall-e-3) are text-to-image models developed by `Ollama`
> using deep learning methodologies to generate digital images from natural language descriptions,
> called "prompts".


See a [usage example](/docs/integrations/tools/dalle_image_generator).
"""
logger.info("## Tools")


"""
## Adapter

See a [usage example](/docs/integrations/adapters/ollama).
"""
logger.info("## Adapter")


"""
## Tokenizer

There are several places you can use the `tiktoken` tokenizer. By default, it is used to count tokens
for Ollama LLMs.

You can also use it to count tokens when splitting documents with
"""
logger.info("## Tokenizer")

CharacterTextSplitter.from_tiktoken_encoder(...)

"""
For a more detailed walkthrough of this, see [this notebook](/docs/how_to/split_by_token/#tiktoken)

## Chain

See a [usage example](https://python.langchain.com/v0.1/docs/guides/productionization/safety/moderation).
"""
logger.info("## Chain")


logger.info("\n\n[DONE]", bright=True)