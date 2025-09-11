from jet.logger import logger
from langchain_community.document_loaders import TrelloLoader
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
# Trello

>[Trello](https://www.atlassian.com/software/trello) is a web-based project management and collaboration tool that allows individuals and teams to organize and track their tasks and projects. It provides a visual interface known as a "board" where users can create lists and cards to represent their tasks and activities.
>The TrelloLoader allows us to load cards from a `Trello` board.


## Installation and Setup
"""
logger.info("# Trello")

pip install py-trello beautifulsoup4

"""
See [setup instructions](/docs/integrations/document_loaders/trello).


## Document Loader

See a [usage example](/docs/integrations/document_loaders/trello).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)