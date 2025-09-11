from jet.logger import logger
from langchain_community.llms.manifest import ManifestWrapper
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
# Hazy Research

This page covers how to use the Hazy Research ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Hazy Research wrappers.

## Installation and Setup
- To use the `manifest`, install it with `pip install manifest-ml`

## Wrappers

### LLM

There exists an LLM wrapper around Hazy Research's `manifest` library.
`manifest` is a python library which is itself a wrapper around many model providers, and adds in caching, history, and more.

To use this wrapper:
"""
logger.info("# Hazy Research")


logger.info("\n\n[DONE]", bright=True)