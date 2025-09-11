from bs4 import BeautifulSoup
from jet.logger import logger
from langchain_community.document_loaders.sitemap import SitemapLoader
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
# Sitemap

Extends from the `WebBaseLoader`, `SitemapLoader` loads a sitemap from a given URL, and then scrapes and loads all pages in the sitemap, returning each page as a Document.

The scraping is done concurrently. There are reasonable limits to concurrent requests, defaulting to 2 per second.  If you aren't concerned about being a good citizen, or you control the scrapped server, or don't care about load you can increase this limit. Note, while this will speed up the scraping process, it may cause the server to block you. Be careful!

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/web_loaders/sitemap/)|
| :--- | :--- | :---: | :---: |  :---: |
| [SiteMapLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.sitemap.SitemapLoader.html#langchain_community.document_loaders.sitemap.SitemapLoader) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ✅ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| SiteMapLoader | ✅ | ❌ | 

## Setup

To access SiteMap document loader you'll need to install the `langchain-community` integration package.

### Credentials

No credentials are needed to run this.

T
o
 
e
n
a
b
l
e
 
a
u
t
o
m
a
t
e
d
 
t
r
a
c
i
n
g
 
o
f
 
y
o
u
r
 
m
o
d
e
l
 
c
a
l
l
s
,
 
s
e
t
 
y
o
u
r
 
[
L
a
n
g
S
m
i
t
h
]
(
h
t
t
p
s
:
/
/
d
o
c
s
.
s
m
i
t
h
.
l
a
n
g
c
h
a
i
n
.
c
o
m
/
)
 
A
P
I
 
k
e
y
:
"""
logger.info("# Sitemap")



"""
### Installation

Install **langchain-community**.
"""
logger.info("### Installation")

# %pip install -qU langchain-community

"""
### Fix notebook asyncio bug
"""
logger.info("### Fix notebook asyncio bug")

# import nest_asyncio

# nest_asyncio.apply()

"""
## Initialization

Now we can instantiate our model object and load documents:
"""
logger.info("## Initialization")


sitemap_loader = SitemapLoader(web_path="https://api.python.langchain.com/sitemap.xml")

"""
## Load
"""
logger.info("## Load")

docs = sitemap_loader.load()
docs[0]

logger.debug(docs[0].metadata)

"""
You can change the `requests_per_second` parameter to increase the max concurrent requests. and use `requests_kwargs` to pass kwargs when send requests.
"""
logger.info("You can change the `requests_per_second` parameter to increase the max concurrent requests. and use `requests_kwargs` to pass kwargs when send requests.")

sitemap_loader.requests_per_second = 2
sitemap_loader.requests_kwargs = {"verify": False}

"""
## Lazy Load

You can also load the pages lazily in order to minimize the memory load.
"""
logger.info("## Lazy Load")

page = []
for doc in sitemap_loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:

        page = []

"""
## Filtering sitemap URLs

Sitemaps can be massive files, with thousands of URLs.  Often you don't need every single one of them.  You can filter the URLs by passing a list of strings or regex patterns to the `filter_urls` parameter.  Only URLs that match one of the patterns will be loaded.
"""
logger.info("## Filtering sitemap URLs")

loader = SitemapLoader(
    web_path="https://api.python.langchain.com/sitemap.xml",
    filter_urls=["https://api.python.langchain.com/en/latest"],
)
documents = loader.load()

documents[0]

"""
## Add custom scraping rules

The `SitemapLoader` uses `beautifulsoup4` for the scraping process, and it scrapes every element on the page by default. The `SitemapLoader` constructor accepts a custom scraping function. This feature can be helpful to tailor the scraping process to your specific needs; for example, you might want to avoid scraping headers or navigation elements.

 The following example shows how to develop and use a custom function to avoid navigation and header elements.

Import the `beautifulsoup4` library and define the custom function.
"""
logger.info("## Add custom scraping rules")

pip install beautifulsoup4



def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")

    for element in nav_elements + header_elements:
        element.decompose()

    return str(content.get_text())

"""
Add your custom function to the `SitemapLoader` object.
"""
logger.info("Add your custom function to the `SitemapLoader` object.")

loader = SitemapLoader(
    "https://api.python.langchain.com/sitemap.xml",
    filter_urls=["https://api.python.langchain.com/en/latest/"],
    parsing_function=remove_nav_and_header_elements,
)

"""
## Local Sitemap

The sitemap loader can also be used to load local files.
"""
logger.info("## Local Sitemap")

sitemap_loader = SitemapLoader(web_path="example_data/sitemap.xml", is_local=True)

docs = sitemap_loader.load()

"""
## API reference

For detailed documentation of all SiteMapLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.sitemap.SitemapLoader.html#langchain_community.document_loaders.sitemap.SitemapLoader
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)