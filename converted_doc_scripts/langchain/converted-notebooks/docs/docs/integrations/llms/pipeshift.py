from jet.logger import logger
from langchain_pipeshift import Pipeshift
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
sidebar_label: Pipeshift
---

# Pipeshift

This will help you get started with Pipeshift completion models (LLMs) using LangChain. For detailed documentation on `Pipeshift` features and configuration options, please refer to the [API reference](https://dashboard.pipeshift.com/docs).

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/llms/pipeshift) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [Pipeshift](https://dashboard.pipeshift.com/docs) | [langchain-pipeshift](https://pypi.org/project/langchain-pipeshift/) | ❌ | - | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-pipeshift?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-pipeshift?style=flat-square&label=%20) |

## Setup

To access Pipeshift models you'll need to create a Pipeshift account, get an API key, and install the `langchain-pipeshift` integration package.

### Credentials

Head to [Pipeshift](https://dashboard.pipeshift.com) to sign up to Pipeshift and generate an API key. Once you've done this set the PIPESHIFT_API_KEY environment variable:
"""
logger.info("# Pipeshift")

# import getpass

if not os.getenv("PIPESHIFT_API_KEY"):
#     os.environ["PIPESHIFT_API_KEY"] = getpass.getpass("Enter your Pipeshift API key: ")

"""
If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

The LangChain Pipeshift integration lives in the `langchain-pipeshift` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-pipeshift

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = Pipeshift(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature=0,
    max_tokens=512,
)

"""
## Invocation
"""
logger.info("## Invocation")

input_text = "Pipeshift is an AI company that "

completion = llm.invoke(input_text)
completion

"""
## Chaining

We can also [chain](/docs/how_to/sequence/) our llm with a prompt template

## API reference

For detailed documentation of all `Pipeshift` features and configurations head to the API reference: https://dashboard.pipeshift.com/docs
"""
logger.info("## Chaining")

logger.info("\n\n[DONE]", bright=True)