from jet.logger import logger
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
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

[FireCrawl](https://firecrawl.dev/?ref=langchain) crawls and convert any website into LLM-ready data. It crawls all accessible subpages and give you clean markdown and metadata for each. No sitemap required.

FireCrawl handles complex tasks such as reverse proxies, caching, rate limits, and content blocked by JavaScript. Built by the [mendable.ai](https://mendable.ai) team.

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/web_loaders/firecrawl/)|
| :--- | :--- | :---: | :---: |  :---: |
| [FireCrawlLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.firecrawl.FireCrawlLoader.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ✅ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| FireCrawlLoader | ✅ | ❌ |

## Setup
"""
logger.info("# FireCrawl")

pip install firecrawl-py

"""
## Usage

You will need to get your own API key. See https://firecrawl.dev
"""
logger.info("## Usage")


loader = FireCrawlLoader(
    url="https://firecrawl.dev", mode="scrape"
)

pages = []
for doc in loader.lazy_load():
    pages.append(doc)
    if len(pages) >= 10:

        pages = []

pages

"""
## Modes

- `scrape`: Scrape single url and return the markdown.
- `crawl`: Crawl the url and all accessible sub pages and return the markdown for each one.
- `map`: Maps the URL and returns a list of semantically related pages.

### Crawl
"""
logger.info("## Modes")

loader = FireCrawlLoader(
    url="https://firecrawl.dev",
    mode="crawl",
)

data = loader.load()

logger.debug(pages[0].page_content[:100])
logger.debug(pages[0].metadata)

"""
#### Crawl Options

You can also pass `params` to the loader. This is a dictionary of options to pass to the crawler. See the [FireCrawl API documentation](https://github.com/mendableai/firecrawl-py) for more information.

### Map
"""
logger.info("#### Crawl Options")

loader = FireCrawlLoader(url="firecrawl.dev", mode="map")

docs = loader.load()

docs

"""
#### Map Options

You can also pass `params` to the loader. This is a dictionary of options to pass to the loader. See the [FireCrawl API documentation](https://github.com/mendableai/firecrawl-py) for more information.

## API reference

For detailed documentation of all `FireCrawlLoader` features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.firecrawl.FireCrawlLoader.html
"""
logger.info("#### Map Options")

logger.info("\n\n[DONE]", bright=True)