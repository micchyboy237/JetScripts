from jet.logger import logger
from langchain_community.llms import ForefrontAI
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
# Forefront AI

> [Forefront AI](https://forefront.ai/) is a platform enabling you to
> fine-tune and inference open-source text generation models


## Installation and Setup

Get an `ForefrontAI` API key
visiting [this page](https://accounts.forefront.ai/sign-in?redirect_url=https%3A%2F%2Fforefront.ai%2Fapp%2Fapi-keys).
 and set it as an environment variable (`FOREFRONTAI_API_KEY`).

## LLM

See a [usage example](/docs/integrations/llms/forefrontai).
"""
logger.info("# Forefront AI")


logger.info("\n\n[DONE]", bright=True)