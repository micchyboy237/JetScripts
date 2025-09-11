from jet.logger import logger
from langchain_community.document_loaders import FigmaFileLoader
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
# Figma

>[Figma](https://www.figma.com/) is a collaborative web application for interface design.

## Installation and Setup

The Figma API requires an `access token`, `node_ids`, and a `file key`.

The `file key` can be pulled from the URL.  https://www.figma.com/file/\{filekey\}/sampleFilename

`Node IDs` are also available in the URL. Click on anything and look for the '?node-id=\{node_id\}' param.

`Access token` [instructions](https://help.figma.com/hc/en-us/articles/8085703771159-Manage-personal-access-tokens).

## Document Loader

See a [usage example](/docs/integrations/document_loaders/figma).
"""
logger.info("# Figma")


logger.info("\n\n[DONE]", bright=True)