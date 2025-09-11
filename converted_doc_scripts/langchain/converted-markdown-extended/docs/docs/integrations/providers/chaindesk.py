from jet.logger import logger
from langchain.retrievers import ChaindeskRetriever
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
# Chaindesk

>[Chaindesk](https://chaindesk.ai) is an [open-source](https://github.com/gmpetrov/databerry) document retrieval platform that helps to connect your personal data with Large Language Models.


## Installation and Setup

We need to sign up for Chaindesk, create a datastore, add some data and get your datastore api endpoint url.
We need the [API Key](https://docs.chaindesk.ai/api-reference/authentication).

## Retriever

See a [usage example](/docs/integrations/retrievers/chaindesk).
"""
logger.info("# Chaindesk")


logger.info("\n\n[DONE]", bright=True)