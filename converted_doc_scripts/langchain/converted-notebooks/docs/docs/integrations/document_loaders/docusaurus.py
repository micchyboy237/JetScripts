from bs4 import BeautifulSoup
from jet.logger import logger
from langchain_community.document_loaders import DocusaurusLoader
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
# Docusaurus
> [Docusaurus](https://docusaurus.io/) is a static-site generator which provides out-of-the-box documentation features.

By utilizing the existing `SitemapLoader`, this loader scans and loads all pages from a given Docusaurus application and returns the main documentation content of each page as a Document.
"""
logger.info("# Docusaurus")


"""
Install necessary dependencies
"""
logger.info("Install necessary dependencies")

# %pip install --upgrade --quiet beautifulsoup4 lxml

# import nest_asyncio

# nest_asyncio.apply()

loader = DocusaurusLoader("https://python.langchain.com")

docs = loader.load()

"""
> `SitemapLoader` also provides the ability to utilize and tweak concurrency which can help optimize the time it takes to load the source documentation. Refer to the [sitemap docs](/docs/integrations/document_loaders/sitemap) for more info.
"""

docs[0]

"""
## Filtering sitemap URLs

Sitemaps can contain thousands of URLs and ften you don't need every single one of them. You can filter the URLs by passing a list of strings or regex patterns to the `url_filter` parameter.  Only URLs that match one of the patterns will be loaded.
"""
logger.info("## Filtering sitemap URLs")

loader = DocusaurusLoader(
    "https://python.langchain.com",
    filter_urls=[
        "https://python.langchain.com/docs/integrations/document_loaders/sitemap"
    ],
)
documents = loader.load()

documents[0]

"""
## Add custom scraping rules

By default, the parser **removes** all but the main content of the docusaurus page, which is normally the `<article>` tag. You also have the option  to define an **inclusive** list HTML tags by providing them as a list utilizing the `custom_html_tags` parameter. For example:
"""
logger.info("## Add custom scraping rules")

loader = DocusaurusLoader(
    "https://python.langchain.com",
    filter_urls=[
        "https://python.langchain.com/docs/integrations/document_loaders/sitemap"
    ],
    custom_html_tags=["#content", ".main"],
)

"""
You can also define an entirely custom parsing function if you need finer-grained control over the returned content for each page.

The following example shows how to develop and use a custom function to avoid navigation and header elements.
"""
logger.info("You can also define an entirely custom parsing function if you need finer-grained control over the returned content for each page.")



def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")

    for element in nav_elements + header_elements:
        element.decompose()

    return str(content.get_text())

"""
Add your custom function to the `DocusaurusLoader` object.
"""
logger.info("Add your custom function to the `DocusaurusLoader` object.")

loader = DocusaurusLoader(
    "https://python.langchain.com",
    filter_urls=[
        "https://python.langchain.com/docs/integrations/document_loaders/sitemap"
    ],
    parsing_function=remove_nav_and_header_elements,
)

logger.info("\n\n[DONE]", bright=True)