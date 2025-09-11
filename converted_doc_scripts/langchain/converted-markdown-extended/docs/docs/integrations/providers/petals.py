from jet.logger import logger
from langchain_community.llms import Petals
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
# Petals

This page covers how to use the Petals ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Petals wrappers.

## Installation and Setup
- Install with `pip install petals`
- Get a Hugging Face api key and set it as an environment variable (`HUGGINGFACE_API_KEY`)

## Wrappers

### LLM

There exists an Petals LLM wrapper, which you can access with
"""
logger.info("# Petals")


logger.info("\n\n[DONE]", bright=True)