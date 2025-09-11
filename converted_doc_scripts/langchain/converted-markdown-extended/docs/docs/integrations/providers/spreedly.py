from jet.logger import logger
from langchain_community.document_loaders import SpreedlyLoader
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
# Spreedly

>[Spreedly](https://docs.spreedly.com/) is a service that allows you to securely store credit cards and use them to transact against any number of payment gateways and third party APIs. It does this by simultaneously providing a card tokenization/vault service as well as a gateway and receiver integration service. Payment methods tokenized by Spreedly are stored at `Spreedly`, allowing you to independently store a card and then pass that card to different end points based on your business requirements.

## Installation and Setup

See [setup instructions](/docs/integrations/document_loaders/spreedly).

## Document Loader

See a [usage example](/docs/integrations/document_loaders/spreedly).
"""
logger.info("# Spreedly")


logger.info("\n\n[DONE]", bright=True)