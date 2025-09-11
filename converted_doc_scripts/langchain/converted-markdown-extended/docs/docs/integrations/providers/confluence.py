from jet.logger import logger
from langchain_community.document_loaders import ConfluenceLoader
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
# Confluence

>[Confluence](https://www.atlassian.com/software/confluence) is a wiki collaboration platform that saves and organizes all of the project-related material. `Confluence` is a knowledge base that primarily handles content management activities.


## Installation and Setup
"""
logger.info("# Confluence")

pip install atlassian-python-api

"""
We need to set up `username/api_key` or `Oauth2 login`.
See [instructions](https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/).


## Document Loader

See a [usage example](/docs/integrations/document_loaders/confluence).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)