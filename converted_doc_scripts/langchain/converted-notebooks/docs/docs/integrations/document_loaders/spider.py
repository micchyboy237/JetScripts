from jet.logger import logger
from langchain_community.document_loaders import SpiderLoader
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
# Spider
[Spider](https://spider.cloud/) is the [fastest](https://github.com/spider-rs/spider/blob/main/benches/BENCHMARKS.md) and most affordable crawler and scraper that returns LLM-ready data.

## Setup
"""
logger.info("# Spider")

pip install spider-client

"""
## Usage
To use spider you need to have an API key from [spider.cloud](https://spider.cloud/).
"""
logger.info("## Usage")


loader = SpiderLoader(
    url="https://spider.cloud",
    mode="scrape",  # if no API key is provided it looks for SPIDER_API_KEY in env
)

data = loader.load()
logger.debug(data)

"""
## Modes
- `scrape`: Default mode that scrapes a single URL
- `crawl`: Crawl all subpages of the domain url provided

## Crawler options
The `params` parameter is a dictionary that can be passed to the loader. See the [Spider documentation](https://spider.cloud/docs/api) to see all available parameters
"""
logger.info("## Modes")

logger.info("\n\n[DONE]", bright=True)