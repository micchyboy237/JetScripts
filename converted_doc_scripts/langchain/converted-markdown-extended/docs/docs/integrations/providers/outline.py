from jet.logger import logger
from langchain.retrievers import OutlineRetriever
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
# Outline

> [Outline](https://www.getoutline.com/) is an open-source collaborative knowledge base platform designed for team information sharing.

## Setup

You first need to [create an api key](https://www.getoutline.com/developers#section/Authentication) for your Outline instance. Then you need to set the following environment variables:
"""
logger.info("# Outline")


os.environ["OUTLINE_API_KEY"] = "xxx"
os.environ["OUTLINE_INSTANCE_URL"] = "https://app.getoutline.com"

"""
## Retriever

See a [usage example](/docs/integrations/retrievers/outline).
"""
logger.info("## Retriever")


logger.info("\n\n[DONE]", bright=True)