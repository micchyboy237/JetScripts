from jet.logger import logger
from langchain_community.document_loaders.fauna import FaunaLoader
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
# Fauna

>[Fauna](https://fauna.com/) is a distributed document-relational database
> that combines the flexibility of documents with the power of a relational,
> ACID compliant database that scales across regions, clouds or the globe.


## Installation and Setup

We have to get the secret key.
See the detailed [guide](https://docs.fauna.com/fauna/current/learn/security_model/).

We have to install the `fauna` package.
"""
logger.info("# Fauna")

pip install -U fauna

"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/fauna).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)