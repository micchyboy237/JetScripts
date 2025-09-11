from jet.logger import logger
from langchain_community.chat_models import ChatOctoAI
from langchain_community.embeddings.octoai_embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
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
# OctoAI

>[OctoAI](https://docs.octoai.cloud/docs) offers easy access to efficient compute
> and enables users to integrate their choice of AI models into applications.
> The `OctoAI` compute service helps you run, tune, and scale AI applications easily.


## Installation and Setup

- Install the `ollama` Python package:
  ```bash
#   pip install ollama
  ````
- Register on `OctoAI` and get an API Token from [your OctoAI account page](https://octoai.cloud/settings).


## Chat models

See a [usage example](/docs/integrations/chat/octoai).
"""
logger.info("# OctoAI")


"""
## LLMs

See a [usage example](/docs/integrations/llms/octoai).
"""
logger.info("## LLMs")


"""
## Embedding models
"""
logger.info("## Embedding models")


logger.info("\n\n[DONE]", bright=True)