from jet.logger import logger
from langchain_aimlapi import AimlapiLLM
from langchain_core.prompts import PromptTemplate
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

# AimlapiLLM

This page will help you get started with AI/ML API [text completion models](/docs/concepts/text_llms). For detailed documentation of all AimlapiLLM features and configurations, head to the [API reference](https://docs.aimlapi.com/?utm_source=langchain&utm_medium=github&utm_campaign=integration).

AI/ML API provides access to **300+ models** (Deepseek, Gemini, ChatGPT, etc.) via high-uptime and high-rate API.

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| AimlapiLLM | langchain-aimlapi | ✅ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-aimlapi?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-aimlapi?style=flat-square&label=%20) |

### Model features
| Tool calling | Structured output | JSON mode | Image input | Audio input | Video input | Token-level streaming | Native async | Token usage | Logprobs |
|:------------:|:-----------------:|:---------:|:-----------:|:-----------:|:-----------:|:---------------------:|:------------:|:-----------:|:--------:|
|      ✅       |         ✅         |     ✅     |      ✅      |      ✅      |      ✅      |           ✅           |      ✅       |      ✅      |    ✅     |

## Setup
To access AI/ML API models, sign up at [aimlapi.com](https://aimlapi.com/app/?utm_source=langchain&utm_medium=github&utm_campaign=integration), generate an API key, and set the `AIMLAPI_API_KEY` environment variable:
"""
logger.info("# AimlapiLLM")

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
Now we can instantiate the `AimlapiLLM` model and generate text completions:
"""
logger.info("## Instantiation")


llm = AimlapiLLM(
    model="llama3.2",
    temperature=0.5,
    max_tokens=256,
)

"""
## Invocation
You can invoke the model with a prompt:
"""
logger.info("## Invocation")

response = llm.invoke("Explain the bubble sort algorithm in Python.")
logger.debug(response)

"""
## Streaming Invocation
You can also stream responses token-by-token:
"""
logger.info("## Streaming Invocation")

llm = AimlapiLLM(
    model="llama3.2",
)

for chunk in llm.stream("List top 5 programming languages in 2025 with reasons."):
    logger.debug(chunk, end="", flush=True)

"""
## API reference

For detailed documentation of all AimlapiLLM features and configurations, visit the [API Reference](https://docs.aimlapi.com/?utm_source=langchain&utm_medium=github&utm_campaign=integration).

## Chaining

You can also easily combine with a prompt template for easy structuring of user input. We can do this using [LCEL](/docs/concepts/lcel)
"""
logger.info("## API reference")


prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm

chain.invoke({"topic": "bears"})

"""
## API reference

For detailed documentation of all `AI/ML API` llm features and configurations head to the API reference: [API Reference](https://docs.aimlapi.com/?utm_source=langchain&utm_medium=github&utm_campaign=integration)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)