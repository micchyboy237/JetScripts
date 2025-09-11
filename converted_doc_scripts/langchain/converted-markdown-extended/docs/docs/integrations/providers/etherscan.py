from jet.logger import logger
from langchain_community.document_loaders import EtherscanLoader
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
# Etherscan

>[Etherscan](https://docs.etherscan.io/) is the leading blockchain explorer,
> search, API and analytics platform for `Ethereum`, a decentralized smart contracts platform.


## Installation and Setup

See the detailed [installation guide](/docs/integrations/document_loaders/etherscan).


## Document Loader

See a [usage example](/docs/integrations/document_loaders/etherscan).
"""
logger.info("# Etherscan")


logger.info("\n\n[DONE]", bright=True)