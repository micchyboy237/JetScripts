from jet.logger import logger
from langchain_community.vectorstores import Tair
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
# Tair

>[Alibaba Cloud Tair](https://www.alibabacloud.com/help/en/tair/latest/what-is-tair) is a cloud native in-memory database service
> developed by `Alibaba Cloud`. It provides rich data models and enterprise-grade capabilities to
> support your real-time online scenarios while maintaining full compatibility with open-source `Redis`.
> `Tair` also introduces persistent memory-optimized instances that are based on
> new non-volatile memory (NVM) storage medium.

## Installation and Setup

Install Tair Python SDK:
"""
logger.info("# Tair")

pip install tair

"""
## Vector Store
"""
logger.info("## Vector Store")


"""
See a [usage example](/docs/integrations/vectorstores/tair).
"""
logger.info("See a [usage example](/docs/integrations/vectorstores/tair).")

logger.info("\n\n[DONE]", bright=True)