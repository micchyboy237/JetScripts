from jet.logger import logger
from langchain_community.llms import NIBittensorLLM
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
# Bittensor

>[Neural Internet Bittensor](https://neuralinternet.ai/) network, an open source protocol
> that powers a decentralized, blockchain-based, machine learning network.

## Installation and Setup

Get your API_KEY from [Neural Internet](https://neuralinternet.ai/).


## LLMs

See a [usage example](/docs/integrations/llms/bittensor).
"""
logger.info("# Bittensor")


logger.info("\n\n[DONE]", bright=True)