from jet.logger import logger
from langchain_community.document_loaders import WebBaseLoader
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
# WebBaseLoader

This covers how to use `WebBaseLoader` to load all text from `HTML` webpages into a document format that we can use downstream. For more custom logic for loading webpages look at some child class examples such as `IMSDbLoader`, `AZLyricsLoader`, and `CollegeConfidentialLoader`. 

If you don't want to worry about website crawling, bypassing JS-blocking sites, and data cleaning, consider using `FireCrawlLoader` or the faster option `SpiderLoader`.

## Overview
### Integration details

- TODO: Fill in table features.
- TODO: Remove JS support link if not relevant, otherwise ensure link is correct.
- TODO: Make sure API reference links are correct.

| Class | Package | Local | Serializable | JS support|
| :--- | :--- | :---: | :---: |  :---: |
| [WebBaseLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| WebBaseLoader | ✅ | ✅ | 

## Setup

### Credentials

`WebBaseLoader` does not require any credentials.

### Installation

To use the `WebBaseLoader` you first need to install the `langchain-community` python package.
"""
logger.info("# WebBaseLoader")

# %pip install -qU langchain-community beautifulsoup4

"""
## Initialization

Now we can instantiate our model object and load documents:
"""
logger.info("## Initialization")


loader = WebBaseLoader("https://www.example.com/")

"""
To bypass SSL verification errors during fetching, you can set the "verify" option:

`loader.requests_kwargs = {'verify':False}`

### Initialization with multiple pages

You can also pass in a list of pages to load from.
"""
logger.info("### Initialization with multiple pages")

loader_multiple_pages = WebBaseLoader(
    ["https://www.example.com/", "https://google.com"]
)

"""
## Load
"""
logger.info("## Load")

docs = loader.load()

docs[0]

logger.debug(docs[0].metadata)

"""
### Load multiple urls concurrently

You can speed up the scraping process by scraping and parsing multiple urls concurrently.

There are reasonable limits to concurrent requests, defaulting to 2 per second.  If you aren't concerned about being a good citizen, or you control the server you are scraping and don't care about load, you can change the `requests_per_second` parameter to increase the max concurrent requests.  Note, while this will speed up the scraping process, but may cause the server to block you.  Be careful!
"""
logger.info("### Load multiple urls concurrently")

# %pip install -qU  nest_asyncio

# import nest_asyncio

# nest_asyncio.apply()

loader = WebBaseLoader(["https://www.example.com/", "https://google.com"])
loader.requests_per_second = 1
docs = loader.aload()
docs

"""
### Loading a xml file, or using a different BeautifulSoup parser

You can also look at `SitemapLoader` for an example of how to load a sitemap file, which is an example of using this feature.
"""
logger.info("### Loading a xml file, or using a different BeautifulSoup parser")

loader = WebBaseLoader(
    "https://www.govinfo.gov/content/pkg/CFR-2018-title10-vol3/xml/CFR-2018-title10-vol3-sec431-86.xml"
)
loader.default_parser = "xml"
docs = loader.load()
docs

"""
## Lazy Load

You can use lazy loading to only load one page at a time in order to minimize memory requirements.
"""
logger.info("## Lazy Load")

pages = []
for doc in loader.lazy_load():
    pages.append(doc)

logger.debug(pages[0].page_content[:100])
logger.debug(pages[0].metadata)

"""
### Async
"""
logger.info("### Async")

pages = []
async for doc in loader.alazy_load():
    pages.append(doc)

logger.debug(pages[0].page_content[:100])
logger.debug(pages[0].metadata)

"""
## Using proxies

Sometimes you might need to use proxies to get around IP blocks. You can pass in a dictionary of proxies to the loader (and `requests` underneath) to use them.
"""
logger.info("## Using proxies")

loader = WebBaseLoader(
    "https://www.walmart.com/search?q=parrots",
    proxies={
        "http": "http://{username}:{password}:@proxy.service.com:6666/",
        "https": "https://{username}:{password}:@proxy.service.com:6666/",
    },
)
docs = loader.load()

"""
## API reference

For detailed documentation of all `WebBaseLoader` features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)