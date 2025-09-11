from jet.logger import logger
from langchain_community.document_loaders import IuguLoader
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
# Iugu

>[Iugu](https://www.iugu.com/) is a Brazilian services and software as a service (SaaS)
> company. It offers payment-processing software and application programming
> interfaces for e-commerce websites and mobile applications.


## Installation and Setup

The `Iugu API` requires an access token, which can be found inside of the `Iugu` dashboard.


## Document Loader

See a [usage example](/docs/integrations/document_loaders/iugu).
"""
logger.info("# Iugu")


logger.info("\n\n[DONE]", bright=True)