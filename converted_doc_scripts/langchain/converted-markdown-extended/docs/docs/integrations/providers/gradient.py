from jet.logger import logger
from langchain_community.embeddings import GradientEmbeddings
from langchain_community.llms import GradientLLM
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
# Gradient

>[Gradient](https://gradient.ai/) allows to fine tune and get completions on LLMs with a simple web API.

## Installation and Setup
- Install the Python SDK :
"""
logger.info("# Gradient")

pip install gradientai

"""
Get a [Gradient access token and workspace](https://gradient.ai/) and set it as an environment variable (`Gradient_ACCESS_TOKEN`) and (`GRADIENT_WORKSPACE_ID`)

## LLM

There exists an Gradient LLM wrapper, which you can access with
See a [usage example](/docs/integrations/llms/gradient).
"""
logger.info("## LLM")


"""
## Text Embedding Model

There exists an Gradient Embedding model, which you can access with
"""
logger.info("## Text Embedding Model")


"""
For a more detailed walkthrough of this, see [this notebook](/docs/integrations/text_embedding/gradient)
"""
logger.info("For a more detailed walkthrough of this, see [this notebook](/docs/integrations/text_embedding/gradient)")

logger.info("\n\n[DONE]", bright=True)