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

The TrelloLoader allows you to load cards from a Trello board and is implemented on top of [py-trello](https://pypi.org/project/py-trello/)

This currently supports `api_key/token` only.

1. Credentials generation: https://trello.com/power-ups/admin/

2. Click in the manual token generation link to get the token.

To specify the API key and token you can either set the environment variables ``TRELLO_API_KEY`` and ``TRELLO_TOKEN`` or you can pass ``api_key`` and ``token`` directly into the `from_credentials` convenience constructor method.

This loader allows you to provide the board name to pull in the corresponding cards into Document objects.

Notice that the board "name" is also called "title" in oficial documentation:

https://support.atlassian.com/trello/docs/changing-a-boards-title-and-description/

You can also specify several load parameters to include / remove different fields both from the document page_content properties and metadata.

## Features
- Load cards from a Trello board.
- Filter cards based on their status (open or closed).
- Include card names, comments, and checklists in the loaded documents.
- Customize the additional metadata fields to include in the document.

By default all card fields are included for the full text page_content and metadata accordinly.
"""
logger.info("# Trello")

# %pip install --upgrade --quiet  py-trello beautifulsoup4 lxml

# from getpass import getpass

# API_KEY = getpass()
# TOKEN = getpass()


loader = TrelloLoader.from_credentials(
    "Awesome Board",
    api_key=API_KEY,
    token=TOKEN,
    card_filter="open",
)
documents = loader.load()

logger.debug(documents[0].page_content)
logger.debug(documents[0].metadata)

loader = TrelloLoader.from_credentials(
    "Awesome Board",
    api_key=API_KEY,
    token=TOKEN,
    extra_metadata=("list"),
)
documents = loader.load()

logger.debug(documents[0].page_content)
logger.debug(documents[0].metadata)

loader = TrelloLoader.from_credentials(
    "test",
    api_key=API_KEY,
    token=TOKEN,
    include_card_name=False,
    include_checklist=False,
    include_comments=False,
)
documents = loader.load()

logger.debug("Document: " + documents[0].page_content)
logger.debug(documents[0].metadata)

logger.info("\n\n[DONE]", bright=True)