from jet.logger import logger
from langchain_community.document_loaders import FireCrawlLoader
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
# FireCrawl

>[FireCrawl](https://firecrawl.dev/?ref=langchain) crawls and converts any website into LLM-ready data.
> It crawls all accessible subpages and give you clean markdown
> and metadata for each. No sitemap required.


## Installation and Setup

Install the python SDK:
"""
logger.info("# FireCrawl")

pip install firecrawl-py==0.0.20

"""
## Document loader

See a [usage example](/docs/integrations/document_loaders/firecrawl).
"""
logger.info("## Document loader")


logger.info("\n\n[DONE]", bright=True)