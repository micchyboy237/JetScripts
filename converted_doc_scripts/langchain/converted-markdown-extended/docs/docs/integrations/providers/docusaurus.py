from jet.logger import logger
from langchain_community.document_loaders import DocusaurusLoader
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
# Docusaurus

>[Docusaurus](https://docusaurus.io/) is a static-site generator which provides
> out-of-the-box documentation features.


## Installation and Setup
"""
logger.info("# Docusaurus")

pip install -U beautifulsoup4 lxml

"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/docusaurus).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)