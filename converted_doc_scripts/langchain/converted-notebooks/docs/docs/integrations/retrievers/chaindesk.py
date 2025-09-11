from jet.logger import logger
from langchain_community.retrievers import ChaindeskRetriever
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
# Chaindesk

>[Chaindesk platform](https://docs.chaindesk.ai/introduction) brings data from anywhere (Datsources: Text, PDF, Word, PowerPpoint, Excel, Notion, Airtable, Google Sheets, etc..) into Datastores (container of multiple Datasources).
Then your Datastores can be connected to ChatGPT via Plugins or any other Large Langue Model (LLM) via the `Chaindesk API`.

This notebook shows how to use [Chaindesk's](https://www.chaindesk.ai/) retriever.

First, you will need to sign up for Chaindesk, create a datastore, add some data and get your datastore api endpoint url. You need the [API Key](https://docs.chaindesk.ai/api-reference/authentication).
"""
logger.info("# Chaindesk")



"""
## Query

Now that our index is set up, we can set up a retriever and start querying it.
"""
logger.info("## Query")


retriever = ChaindeskRetriever(
    datastore_url="https://clg1xg2h80000l708dymr0fxc.chaindesk.ai/query",
)

retriever.invoke("What is Daftpage?")

logger.info("\n\n[DONE]", bright=True)