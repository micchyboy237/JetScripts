from jet.logger import logger
from langchain_community.document_loaders import ModernTreasuryLoader
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
# Modern Treasury

>[Modern Treasury](https://www.moderntreasury.com/) simplifies complex payment operations. It is a unified platform to power products and processes that move money.
>- Connect to banks and payment systems
>- Track transactions and balances in real-time
>- Automate payment operations for scale

## Installation and Setup

There isn't any special setup for it.

## Document Loader

See a [usage example](/docs/integrations/document_loaders/modern_treasury).
"""
logger.info("# Modern Treasury")


logger.info("\n\n[DONE]", bright=True)