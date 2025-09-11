from jet.logger import logger
from langchain_core.prompts import PromptTemplate
from langchain_modelscope import ModelScopeEndpoint
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
sidebar_label: ModelScope
---

# ModelScopeEndpoint

ModelScope ([Home](https://www.modelscope.cn/) | [GitHub](https://github.com/modelscope/modelscope)) is built upon the notion of “Model-as-a-Service” (MaaS). It seeks to bring together most advanced machine learning models from the AI community, and streamlines the process of leveraging AI models in real-world applications. The core ModelScope library open-sourced in this repository provides the interfaces and implementations that allow developers to perform model inference, training and evaluation. This will help you get started with ModelScope completion models (LLMs) using LangChain.

## Overview
### Integration details

| Provider  | Class | Package | Local | Serializable | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ModelScope](/docs/integrations/providers/modelscope/) | ModelScopeEndpoint | [langchain-modelscope-integration](https://pypi.org/project/langchain-modelscope-integration/) | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-modelscope-integration?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-modelscope-integration?style=flat-square&label=%20) |


## Setup

To access ModelScope models you'll need to create a ModelScope account, get an SDK token, and install the `langchain-modelscope-integration` integration package.

### Credentials


Head to [ModelScope](https://modelscope.cn/) to sign up to ModelScope and generate an [SDK token](https://modelscope.cn/my/myaccesstoken). Once you've done this set the `MODELSCOPE_SDK_TOKEN` environment variable:
"""
logger.info("# ModelScopeEndpoint")

# import getpass

if not os.getenv("MODELSCOPE_SDK_TOKEN"):
#     os.environ["MODELSCOPE_SDK_TOKEN"] = getpass.getpass(
        "Enter your ModelScope SDK token: "
    )

"""
### Installation

The LangChain ModelScope integration lives in the `langchain-modelscope-integration` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-modelscope-integration

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ModelScopeEndpoint(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0,
    max_tokens=1024,
    timeout=60,
)

"""
## Invocation
"""
logger.info("## Invocation")

input_text = "Write a quick sort algorithm in python"

completion = llm.invoke(input_text)
completion

for chunk in llm.stream("write a python program to sort an array"):
    logger.debug(chunk, end="", flush=True)

"""
## Chaining

We can [chain](/docs/how_to/sequence/) our completion model with a prompt template like so:
"""
logger.info("## Chaining")


prompt = PromptTemplate(template="How to say {input} in {output_language}:\n")

chain = prompt | llm
chain.invoke(
    {
        "output_language": "Chinese",
        "input": "I love programming.",
    }
)

"""
## API reference

Refer to https://modelscope.cn/docs/model-service/API-Inference/intro for more details.
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)