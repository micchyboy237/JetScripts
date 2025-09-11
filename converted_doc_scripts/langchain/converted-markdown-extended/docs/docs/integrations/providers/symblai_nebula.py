from jet.logger import logger
from langchain_community.llms import Nebula
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
# Nebula

This page covers how to use [Nebula](https://symbl.ai/nebula), [Symbl.ai](https://symbl.ai/)'s LLM, ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Nebula wrappers.

## Installation and Setup

- Get an [Nebula API Key](https://info.symbl.ai/Nebula_Private_Beta.html) and set as environment variable `NEBULA_API_KEY`
- Please see the [Nebula documentation](https://docs.symbl.ai/docs/nebula-llm) for more details.

### LLM

There exists an Nebula LLM wrapper, which you can access with
"""
logger.info("# Nebula")

llm = Nebula()

logger.info("\n\n[DONE]", bright=True)