from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
import ollama
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
# Helicone

This page covers how to use the [Helicone](https://helicone.ai) ecosystem within LangChain.

## What is Helicone?

Helicone is an [open-source](https://github.com/Helicone/helicone) observability platform that proxies your Ollama traffic and provides you key insights into your spend, latency and usage.

![Screenshot of the Helicone dashboard showing average requests per day, response time, tokens per response, total cost, and a graph of requests over time.](/img/HeliconeDashboard.png "Helicone Dashboard")

## Quick start

With your LangChain environment you can just add the following parameter.
"""
logger.info("# Helicone")

export OPENAI_API_BASE="https://oai.hconeai.com/v1"

"""
Now head over to [helicone.ai](https://www.helicone.ai/signup) to create your account, and add your Ollama API key within our dashboard to view your logs.

![Interface for entering and managing Ollama API keys in the Helicone dashboard.](/img/HeliconeKeys.png "Helicone API Key Input")

## How to enable Helicone caching
"""
logger.info("## How to enable Helicone caching")

ollama.api_base = "https://oai.hconeai.com/v1"

llm = Ollama(temperature=0.9, headers={"Helicone-Cache-Enabled": "true"})
text = "What is a helicone?"
logger.debug(llm.invoke(text))

"""
[Helicone caching docs](https://docs.helicone.ai/advanced-usage/caching)

## How to use Helicone custom properties
"""
logger.info("## How to use Helicone custom properties")

ollama.api_base = "https://oai.hconeai.com/v1"

llm = Ollama(temperature=0.9, headers={
        "Helicone-Property-Session": "24",
        "Helicone-Property-Conversation": "support_issue_2",
        "Helicone-Property-App": "mobile",
      })
text = "What is a helicone?"
logger.debug(llm.invoke(text))

"""
[Helicone property docs](https://docs.helicone.ai/advanced-usage/custom-properties)
"""

logger.info("\n\n[DONE]", bright=True)