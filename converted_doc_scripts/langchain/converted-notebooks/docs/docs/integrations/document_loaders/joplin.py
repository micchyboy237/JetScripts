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

>[Joplin](https://joplinapp.org/) is an open-source note-taking app. Capture your thoughts and securely access them from any device.

This notebook covers how to load documents from a `Joplin` database.

`Joplin` has a [REST API](https://joplinapp.org/api/references/rest_api/) for accessing its local database. This loader uses the API to retrieve all notes in the database and their metadata. This requires an access token that can be obtained from the app by following these steps:

1. Open the `Joplin` app. The app must stay open while the documents are being loaded.
2. Go to settings / options and select "Web Clipper".
3. Make sure that the Web Clipper service is enabled.
4. Under "Advanced Options", copy the authorization token.

You may either initialize the loader directly with the access token, or store it in the environment variable JOPLIN_ACCESS_TOKEN.

An alternative to this approach is to export the `Joplin`'s note database to Markdown files (optionally, with Front Matter metadata) and use a Markdown loader, such as ObsidianLoader, to load them.
"""
logger.info("# Joplin")


loader = JoplinLoader(access_token="<access-token>")

docs = loader.load()

logger.info("\n\n[DONE]", bright=True)