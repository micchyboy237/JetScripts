from jet.logger import logger
from langchain_community.embeddings import AlephAlphaSymmetricSemanticEmbedding, AlephAlphaAsymmetricSemanticEmbedding
from langchain_community.llms import AlephAlpha
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
# Aleph Alpha

>[Aleph Alpha](https://docs.aleph-alpha.com/) was founded in 2019 with the mission to research and build the foundational technology for an era of strong AI. The team of international scientists, engineers, and innovators researches, develops, and deploys transformative AI like large language and multimodal models and runs the fastest European commercial AI cluster.

>[The Luminous series](https://docs.aleph-alpha.com/docs/introduction/luminous/) is a family of large language models.

## Installation and Setup
"""
logger.info("# Aleph Alpha")

pip install aleph-alpha-client

"""
You have to create a new token. Please, see [instructions](https://docs.aleph-alpha.com/docs/account/#create-a-new-token).
"""
logger.info("You have to create a new token. Please, see [instructions](https://docs.aleph-alpha.com/docs/account/#create-a-new-token).")

# from getpass import getpass

# ALEPH_ALPHA_API_KEY = getpass()

"""
## LLM

See a [usage example](/docs/integrations/llms/aleph_alpha).
"""
logger.info("## LLM")


"""
## Text Embedding Models

See a [usage example](/docs/integrations/text_embedding/aleph_alpha).
"""
logger.info("## Text Embedding Models")


logger.info("\n\n[DONE]", bright=True)