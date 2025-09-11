from jet.adapters.langchain.chat_ollama import ChatOllamaLLM
from jet.logger import logger
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
sidebar_label: Ollama
sidebar_class_name: hidden
---

# OllamaLLM

:::caution
You are currently on a page documenting the use of Ollama legacy Claude 2 models as [text completion models](/docs/concepts/text_llms). The latest and most popular Ollama models are [chat completion models](/docs/concepts/chat_models), and the text completion models have been deprecated.

You are probably looking for [this page instead](/docs/integrations/chat/anthropic/).
:::

This example goes over how to use LangChain to interact with `Ollama` models.

## Installation
"""
logger.info("# OllamaLLM")

# %pip install -qU langchain-anthropic

"""
## Environment Setup

# We'll need to get an [Ollama](https://console.anthropic.com/settings/keys) API key and set the `ANTHROPIC_API_KEY` environment variable:
"""
logger.info("## Environment Setup")

# from getpass import getpass

# if "ANTHROPIC_API_KEY" not in os.environ:
#     os.environ["ANTHROPIC_API_KEY"] = getpass()

"""
## Usage
"""
logger.info("## Usage")


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

model = OllamaLLM(model="claude-2.1")

chain = prompt | model

chain.invoke({"question": "What is LangChain?"})

logger.info("\n\n[DONE]", bright=True)
