from jet.logger import logger
from langchain_community.document_loaders import BrowserlessLoader
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
# Browserless

>[Browserless](https://www.browserless.io/docs/start) is a service that allows you to
> run headless Chrome instances in the cloud. It’s a great way to run browser-based
> automation at scale without having to worry about managing your own infrastructure.

## Installation and Setup

We have to get the API key [here](https://www.browserless.io/pricing/).


## Document loader

See a [usage example](/docs/integrations/document_loaders/browserless).
"""
logger.info("# Browserless")


logger.info("\n\n[DONE]", bright=True)