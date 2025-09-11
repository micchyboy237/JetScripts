from jet.logger import logger
from langchain_hyperbrowser import HyperbrowserLoader
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
# HyperbrowserLoader

[Hyperbrowser](https://hyperbrowser.ai) is a platform for running and scaling headless browsers. It lets you launch and manage browser sessions at scale and provides easy to use solutions for any webscraping needs, such as scraping a single page or crawling an entire site.

Key Features:
- Instant Scalability - Spin up hundreds of browser sessions in seconds without infrastructure headaches
- Simple Integration - Works seamlessly with popular tools like Puppeteer and Playwright
- Powerful APIs - Easy to use APIs for scraping/crawling any site, and much more
- Bypass Anti-Bot Measures - Built-in stealth mode, ad blocking, automatic CAPTCHA solving, and rotating proxies

This notebook provides a quick overview for getting started with Hyperbrowser [document loader](https://python.langchain.com/docs/concepts/#document-loaders).

For more information about Hyperbrowser, please visit the [Hyperbrowser website](https://hyperbrowser.ai) or if you want to check out the docs, you can visit the [Hyperbrowser docs](https://docs.hyperbrowser.ai).

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support|
| :--- | :--- | :---: | :---: |  :---: |
| HyperbrowserLoader | langchain-hyperbrowser | ❌ | ❌ | ❌ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support |
| :---: | :---: | :---: | 
| HyperbrowserLoader | ✅ | ✅ | 

## Setup

To access Hyperbrowser document loader you'll need to install the `langchain-hyperbrowser` integration package, and create a Hyperbrowser account and get an API key.

### Credentials

Head to [Hyperbrowser](https://app.hyperbrowser.ai/) to sign up and generate an API key. Once you've done this set the HYPERBROWSER_API_KEY environment variable:

### Installation

Install **langchain-hyperbrowser**.
"""
logger.info("# HyperbrowserLoader")

# %pip install -qU langchain-hyperbrowser

"""
## Initialization

Now we can instantiate our model object and load documents:
"""
logger.info("## Initialization")


loader = HyperbrowserLoader(
    urls="https://example.com",
)

"""
## Load
"""
logger.info("## Load")

docs = loader.load()
docs[0]

logger.debug(docs[0].metadata)

"""
## Lazy Load
"""
logger.info("## Lazy Load")

page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:

        page = []

"""
## Advanced Usage

You can specify the operation to be performed by the loader. The default operation is `scrape`. For `scrape`, you can provide a single URL or a list of URLs to be scraped. For `crawl`, you can only provide a single URL. The `crawl` operation will crawl the provided page and subpages and return a document for each page.
"""
logger.info("## Advanced Usage")

loader = HyperbrowserLoader(
    urls="https://hyperbrowser.ai", operation="crawl"
)

"""
Optional params for the loader can also be provided in the `params` argument. For more information on the supported params, visit https://docs.hyperbrowser.ai/reference/sdks/python/scrape#start-scrape-job-and-wait or https://docs.hyperbrowser.ai/reference/sdks/python/crawl#start-crawl-job-and-wait.
"""
logger.info("Optional params for the loader can also be provided in the `params` argument. For more information on the supported params, visit https://docs.hyperbrowser.ai/reference/sdks/python/scrape#start-scrape-job-and-wait or https://docs.hyperbrowser.ai/reference/sdks/python/crawl#start-crawl-job-and-wait.")

loader = HyperbrowserLoader(
    urls="https://example.com",
    operation="scrape",
    params={"scrape_options": {"include_tags": ["h1", "h2", "p"]}},
)

"""
## API reference

- [GitHub](https://github.com/hyperbrowserai/langchain-hyperbrowser/)
- [PyPi](https://pypi.org/project/langchain-hyperbrowser/)
- [Hyperbrowser Docs](https://docs.hyperbrowser.ai/)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)