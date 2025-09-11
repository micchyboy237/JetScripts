from jet.logger import logger
from langchain_community.llms import PipelineAI
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
# PipelineAI

This page covers how to use the PipelineAI ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific PipelineAI wrappers.

## Installation and Setup

- Install with `pip install pipeline-ai`
- Get a Pipeline Cloud api key and set it as an environment variable (`PIPELINE_API_KEY`)

## Wrappers

### LLM

There exists a PipelineAI LLM wrapper, which you can access with
"""
logger.info("# PipelineAI")


logger.info("\n\n[DONE]", bright=True)