from jet.logger import logger
from langchain_community.llms import KoboldApiLLM
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
# KoboldAI

>[KoboldAI](https://koboldai.com/) is a free, open-source project that allows users to run AI models locally
> on their own computer.
> It's a browser-based front-end that can be used for writing or role playing with an AI.
>[KoboldAI](https://github.com/KoboldAI/KoboldAI-Client) is a "a browser-based front-end for
> AI-assisted writing with multiple local & remote AI models...".
> It has a public and local API that can be used in LangChain.

## Installation and Setup

Check out the [installation guide](https://github.com/KoboldAI/KoboldAI-Client).

## LLMs

See a [usage example](/docs/integrations/llms/koboldai).
"""
logger.info("# KoboldAI")


logger.info("\n\n[DONE]", bright=True)