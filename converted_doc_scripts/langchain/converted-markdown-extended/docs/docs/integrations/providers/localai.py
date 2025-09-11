from jet.logger import logger
from langchain_localai import LocalAIEmbeddings
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
# LocalAI

>[LocalAI](https://localai.io/) is the free, Open Source Ollama alternative.
> `LocalAI` act as a drop-in replacement REST API that’s compatible with Ollama API
> specifications for local inferencing. It allows you to run LLMs, generate images,
> audio (and not only) locally or on-prem with consumer grade hardware,
> supporting multiple model families and architectures.

:::caution
For proper compatibility, please ensure you are using the `ollama` SDK at version **0.x**.
:::

:::info
`langchain-localai` is a 3rd party integration package for LocalAI. It provides a simple way to use LocalAI services in Langchain.
The source code is available on [Github](https://github.com/mkhludnev/langchain-localai)
:::

## Installation and Setup

We have to install several python packages:
"""
logger.info("# LocalAI")

pip install tenacity ollama

"""
## Embedding models

See a [usage example](/docs/integrations/text_embedding/localai).
"""
logger.info("## Embedding models")


logger.info("\n\n[DONE]", bright=True)