from jet.logger import logger
from langchain_community.document_loaders import BrowserlessLoader
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
# Browserless

Browserless is a service that allows you to run headless Chrome instances in the cloud. It's a great way to run browser-based automation at scale without having to worry about managing your own infrastructure.

To use Browserless as a document loader, initialize a `BrowserlessLoader` instance as shown in this notebook. Note that by default, `BrowserlessLoader` returns the `innerText` of the page's `body` element. To disable this and get the raw HTML, set `text_content` to `False`.
"""
logger.info("# Browserless")


BROWSERLESS_API_TOKEN = "YOUR_BROWSERLESS_API_TOKEN"

loader = BrowserlessLoader(
    api_token=BROWSERLESS_API_TOKEN,
    urls=[
        "https://en.wikipedia.org/wiki/Document_classification",
    ],
    text_content=True,
)

documents = loader.load()

logger.debug(documents[0].page_content[:1000])

logger.info("\n\n[DONE]", bright=True)