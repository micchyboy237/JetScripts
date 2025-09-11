from jet.logger import logger
from langchain_community.document_loaders import BeautifulSoupTransformer
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
# Beautiful Soup

>[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) is a Python package for parsing
> HTML and XML documents (including having malformed markup, i.e. non-closed tags, so named after tag soup).
> It creates a parse tree for parsed pages that can be used to extract data from HTML,[3] which
> is useful for web scraping.

## Installation and Setup
"""
logger.info("# Beautiful Soup")

pip install beautifulsoup4

"""
## Document Transformer

See a [usage example](/docs/integrations/document_transformers/beautiful_soup).
"""
logger.info("## Document Transformer")


logger.info("\n\n[DONE]", bright=True)