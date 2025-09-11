from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_litellm import ChatLiteLLM
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router
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
sidebar_label: LiteLLM
---

# ChatLiteLLM and ChatLiteLLMRouter

[LiteLLM](https://github.com/BerriAI/litellm) is a library that simplifies calling Ollama, Azure, Huggingface, Replicate, etc.

This notebook covers how to get started with using Langchain + the LiteLLM I/O library.

This integration contains two main classes:

- ```ChatLiteLLM```: The main Langchain wrapper for basic usage of LiteLLM ([docs](https://docs.litellm.ai/docs/)).
- ```ChatLiteLLMRouter```: A ```ChatLiteLLM``` wrapper that leverages LiteLLM's Router ([docs](https://docs.litellm.ai/docs/routing)).

## Table of Contents
1. [Overview](#overview)
   - [Integration Details](#integration-details)
   - [Model Features](#model-features)
2. [Setup](#setup)
3. [Credentials](#credentials)
4. [Installation](#installation)
5. [Instantiation](#instantiation)
   - [ChatLiteLLM](#chatlitellm)
   - [ChatLiteLLMRouter](#chatlitellmrouter)
6. [Invocation](#invocation)
7. [Async and Streaming Functionality](#async-and-streaming-functionality)
8. [API Reference](#api-reference)

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support| Package downloads | Package latest |
| :---  | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatLiteLLM](https://python.langchain.com/docs/integrations/chat/litellm/#chatlitellm) | [langchain-litellm](https://pypi.org/project/langchain-litellm/)| ❌ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-litellm?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-litellm?style=flat-square&label=%20) |
| [ChatLiteLLMRouter](https://python.langchain.com/docs/integrations/chat/litellm/#chatlitellmrouter) | [langchain-litellm](https://pypi.org/project/langchain-litellm/)| ❌ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-litellm?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-litellm?style=flat-square&label=%20) |

### Model features
| [Tool calling](https://python.langchain.com/docs/how_to/tool_calling/) | [Structured output](https://python.langchain.com/docs/how_to/structured_output/) | JSON mode | Image input | Audio input | Video input | [Token-level streaming](https://python.langchain.com/docs/integrations/chat/litellm/#chatlitellm-also-supports-async-and-streaming-functionality) | [Native async](https://python.langchain.com/docs/integrations/chat/litellm/#chatlitellm-also-supports-async-and-streaming-functionality) | [Token usage](https://python.langchain.com/docs/how_to/chat_token_usage_tracking/) | [Logprobs](https://python.langchain.com/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |

### Setup
To access ```ChatLiteLLM``` and ```ChatLiteLLMRouter``` models, you'll need to install the `langchain-litellm` package and create an Ollama, Ollama, Azure, Replicate, OpenRouter, Hugging Face, Together AI, or Cohere account. Then, you have to get an API key and export it as an environment variable.

## Credentials

You have to choose the LLM provider you want and sign up with them to get their API key.

### Example - Ollama
# Head to https://console.anthropic.com/ to sign up for Ollama and generate an API key. Once you've done this, set the ANTHROPIC_API_KEY environment variable.


### Example - Ollama
# Head to https://platform.ollama.com/api-keys to sign up for Ollama and generate an API key. Once you've done this, set the OPENAI_API_KEY environment variable.
"""
logger.info("# ChatLiteLLM and ChatLiteLLMRouter")


# os.environ["OPENAI_API_KEY"] = "your-ollama-key"
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

"""
### Installation

The LangChain LiteLLM integration is available in the `langchain-litellm` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-litellm

"""
## Instantiation

### ChatLiteLLM
You can instantiate a ```ChatLiteLLM``` model by providing a ```model``` name [supported by LiteLLM](https://docs.litellm.ai/docs/providers).
"""
logger.info("## Instantiation")


llm = ChatLiteLLM(model="llama3.2", temperature=0.1)

"""
### ChatLiteLLMRouter
You can also leverage LiteLLM's routing capabilities by defining your model list as specified [here](https://docs.litellm.ai/docs/routing).
"""
logger.info("### ChatLiteLLMRouter")


model_list = [
    {
        "model_name": "gpt-4.1",
        "litellm_params": {
            "model": "azure/gpt-4.1",
            "api_key": "<your-api-key>",
            "api_version": "2024-10-21",
            "api_base": "https://<your-endpoint>.ollama.azure.com/",
        },
    },
    {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "azure/gpt-4o",
            "api_key": "<your-api-key>",
            "api_version": "2024-10-21",
            "api_base": "https://<your-endpoint>.ollama.azure.com/",
        },
    },
]
litellm_router = Router(model_list=model_list)
llm = ChatLiteLLMRouter(router=litellm_router, model_name="gpt-4.1", temperature=0.1)

"""
## Invocation
Whether you've instantiated a `ChatLiteLLM` or a `ChatLiteLLMRouter`, you can now use the ChatModel through Langchain's API.
"""
logger.info("## Invocation")

response = await llm.ainvoke(
        "Classify the text into neutral, negative or positive. Text: I think the food was okay. Sentiment:"
    )
logger.success(format_json(response))
logger.debug(response)

"""
## Async and Streaming Functionality
`ChatLiteLLM` and `ChatLiteLLMRouter` also support async and streaming functionality:
"""
logger.info("## Async and Streaming Functionality")

for token in llm.stream("Hello, please explain how antibiotics work"):
    logger.debug(token.text(), end="")

"""
## API reference
For detailed documentation of all `ChatLiteLLM` and `ChatLiteLLMRouter` features and configurations, head to the API reference: https://github.com/Akshay-Dongare/langchain-litellm
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)