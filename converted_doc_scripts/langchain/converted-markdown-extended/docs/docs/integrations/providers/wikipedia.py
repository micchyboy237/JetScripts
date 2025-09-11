from jet.logger import logger
from langchain.retrievers import WikipediaRetriever
from langchain_community.document_loaders import WikipediaLoader
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
# Wikipedia

>[Wikipedia](https://wikipedia.org/) is a multilingual free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing system called MediaWiki. `Wikipedia` is the largest and most-read reference work in history.


## Installation and Setup
"""
logger.info("# Wikipedia")

pip install wikipedia

"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/wikipedia).
"""
logger.info("## Document Loader")


"""
## Retriever

See a [usage example](/docs/integrations/retrievers/wikipedia).
"""
logger.info("## Retriever")


logger.info("\n\n[DONE]", bright=True)