from jet.logger import logger
from langchain_community.callbacks import ArizeCallbackHandler
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
# Arize

[Arize](https://arize.com) is an AI observability and LLM evaluation platform that offers
support for LangChain applications, providing detailed traces of input, embeddings, retrieval,
functions, and output messages.


## Installation and Setup

First, you need to install `arize` python package.
"""
logger.info("# Arize")

pip install arize

"""
Second, you need to set up your [Arize account](https://app.arize.com/auth/join)
and get your  `API_KEY` or `SPACE_KEY`.


## Callback handler
"""
logger.info("## Callback handler")


logger.info("\n\n[DONE]", bright=True)