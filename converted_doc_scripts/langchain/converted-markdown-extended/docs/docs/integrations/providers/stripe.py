from jet.logger import logger
from langchain_community.document_loaders import StripeLoader
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
# Stripe

>[Stripe](https://stripe.com/en-ca) is an Irish-American financial services and software as a service (SaaS) company. It offers payment-processing software and application programming interfaces for e-commerce websites and mobile applications.


## Installation and Setup

See [setup instructions](/docs/integrations/document_loaders/stripe).

## Document Loader

See a [usage example](/docs/integrations/document_loaders/stripe).
"""
logger.info("# Stripe")


logger.info("\n\n[DONE]", bright=True)