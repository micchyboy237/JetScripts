from jet.logger import logger
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import DuckDuckGoSearchRun
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
# DuckDuckGo Search

>[DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) is a package that
> searches for words, documents, images, videos, news, maps and text
> translation using the `DuckDuckGo.com` search engine. It is downloading files
> and images to a local hard drive.

## Installation and Setup

You have to install a python package:
"""
logger.info("# DuckDuckGo Search")

pip install duckduckgo-search

"""
## Tools

See a [usage example](/docs/integrations/tools/ddg).

There are two tools available:
"""
logger.info("## Tools")


logger.info("\n\n[DONE]", bright=True)