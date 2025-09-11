from jet.logger import logger
from langchain_community.llms import SelfHostedPipeline, SelfHostedHuggingFaceLLM
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
# Runhouse

This page covers how to use the [Runhouse](https://github.com/run-house/runhouse) ecosystem within LangChain.
It is broken into three parts: installation and setup, LLMs, and Embeddings.

## Installation and Setup
- Install the Python SDK with `pip install runhouse`
- If you'd like to use on-demand cluster, check your cloud credentials with `sky check`

## Self-hosted LLMs
For a basic self-hosted LLM, you can use the `SelfHostedHuggingFaceLLM` class. For more
custom LLMs, you can use the `SelfHostedPipeline` parent class.
"""
logger.info("# Runhouse")


"""
For a more detailed walkthrough of the Self-hosted LLMs, see [this notebook](/docs/integrations/llms/runhouse)

## Self-hosted Embeddings
There are several ways to use self-hosted embeddings with LangChain via Runhouse.

For a basic self-hosted embedding from a Hugging Face Transformers model, you can use
the `SelfHostedEmbedding` class.
"""
logger.info("## Self-hosted Embeddings")


"""
For a more detailed walkthrough of the Self-hosted Embeddings, see [this notebook](/docs/integrations/text_embedding/self-hosted)
"""
logger.info("For a more detailed walkthrough of the Self-hosted Embeddings, see [this notebook](/docs/integrations/text_embedding/self-hosted)")

logger.info("\n\n[DONE]", bright=True)