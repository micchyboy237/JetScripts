from jet.logger import logger
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain_community.vectorstores import AtlasDB
from langchain_nomic import OllamaEmbeddings
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
# Nomic

>[Nomic](https://www.nomic.ai/) builds tools that enable everyone to interact with AI scale datasets and run AI models on consumer computers.
>
>`Nomic` currently offers two products:
>
>- `Atlas`: the Visual Data Engine
>- `GPT4All`: the Open Source Edge Language Model Ecosystem

The Nomic integration exists in two partner packages: [langchain-nomic](https://pypi.org/project/langchain-nomic/)
and in [langchain-community](https://pypi.org/project/langchain-community/).

## Installation

You can install them with:
"""
logger.info("# Nomic")

pip install -U langchain-nomic
pip install -U langchain-community

"""
## LLMs

### GPT4All

See [a usage example](/docs/integrations/llms/gpt4all).
"""
logger.info("## LLMs")


"""
## Embedding models

### OllamaEmbeddings

See [a usage example](/docs/integrations/text_embedding/nomic).
"""
logger.info("## Embedding models")


"""
### GPT4All

See [a usage example](/docs/integrations/text_embedding/gpt4all).
"""
logger.info("### GPT4All")


"""
## Vector store

### Atlas

See [a usage example and installation instructions](/docs/integrations/vectorstores/atlas).
"""
logger.info("## Vector store")


logger.info("\n\n[DONE]", bright=True)