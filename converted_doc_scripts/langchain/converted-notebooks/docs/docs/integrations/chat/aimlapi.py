from jet.logger import logger
from langchain_aimlapi import ChatAimlapi
from langchain_core.prompts import ChatPromptTemplate
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
sidebar_label: AI/ML API
---

# ChatAimlapi

This page will help you get started with AI/ML API [chat models](/docs/concepts/chat_models.mdx). For detailed documentation of all ChatAimlapi features and configurations, head to the [API reference](https://docs.aimlapi.com/?utm_source=langchain&utm_medium=github&utm_campaign=integration).

AI/ML API provides access to **300+ models** (Deepseek, Gemini, ChatGPT, etc.) via high-uptime and high-rate API.

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| ChatAimlapi | langchain-aimlapi | ✅ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-aimlapi?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-aimlapi?style=flat-square&label=%20) |

### Model features
| Tool calling | Structured output | JSON mode | Image input | Audio input | Video input | Token-level streaming | Native async | Token usage | Logprobs |
|:------------:|:-----------------:|:---------:|:-----------:|:-----------:|:-----------:|:---------------------:|:------------:|:-----------:|:--------:|
|      ✅       |         ✅         |     ✅     |      ✅      |      ✅      |      ✅      |           ✅           |      ✅       |      ✅      |    ✅     |

## Setup
To access AI/ML API models, sign up at [aimlapi.com](https://aimlapi.com/app/?utm_source=langchain&utm_medium=github&utm_campaign=integration), generate an API key, and set the `AIMLAPI_API_KEY` environment variable:
"""
logger.info("# ChatAimlapi")

# import getpass

if "AIMLAPI_API_KEY" not in os.environ:
#     os.environ["AIMLAPI_API_KEY"] = getpass.getpass("Enter your AI/ML API key: ")

"""
### Installation
Install the `langchain-aimlapi` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-aimlapi

"""
## Instantiation
Now we can instantiate the `ChatAimlapi` model and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatAimlapi(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.7,
    max_tokens=512,
    timeout=30,
    max_retries=3,
)

"""
## Invocation
You can invoke the model with a list of messages:
"""
logger.info("## Invocation")

messages = [
    ("system", "You are a helpful assistant that translates English to French."),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(messages)
logger.debug(ai_msg.content)

"""
## Chaining
We can chain the model with a prompt template as follows:
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
response = chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)
logger.debug(response.content)

"""
## API reference

For detailed documentation of all ChatAimlapi features and configurations, visit the [API Reference](https://docs.aimlapi.com/?utm_source=langchain&utm_medium=github&utm_campaign=integration).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)