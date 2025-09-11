from jet.logger import logger
from langchain_community.agent_toolkits.ainetwork.toolkit import AINetworkToolkit
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
# AINetwork

>[AI Network](https://www.ainetwork.ai/build-on-ain) is a layer 1 blockchain designed to accommodate
> large-scale AI models, utilizing a decentralized GPU network powered by the
> [$AIN token](https://www.ainetwork.ai/token), enriching AI-driven `NFTs` (`AINFTs`).


## Installation and Setup

You need to install `ain-py` python package.
"""
logger.info("# AINetwork")

pip install ain-py

"""
You need to set the `AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY` environmental variable to your AIN Blockchain Account Private Key.
## Toolkit

See a [usage example](/docs/integrations/tools/ainetwork).
"""
logger.info("## Toolkit")


logger.info("\n\n[DONE]", bright=True)