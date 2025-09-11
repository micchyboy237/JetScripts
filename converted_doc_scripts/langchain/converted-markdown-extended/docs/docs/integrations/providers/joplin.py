from jet.logger import logger
from langchain_community.document_loaders import JoplinLoader
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
# Joplin

>[Joplin](https://joplinapp.org/) is an open-source note-taking app. It captures your thoughts
> and securely accesses them from any device.


## Installation and Setup

The `Joplin API` requires an access token.
You can find installation instructions [here](https://joplinapp.org/api/references/rest_api/).


## Document Loader

See a [usage example](/docs/integrations/document_loaders/joplin).
"""
logger.info("# Joplin")


logger.info("\n\n[DONE]", bright=True)