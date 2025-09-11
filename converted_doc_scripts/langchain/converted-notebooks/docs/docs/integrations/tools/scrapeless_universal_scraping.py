from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_scrapeless import ScrapelessUniversalScrapingTool
from langgraph.prebuilt import create_react_agent
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
# Scrapeless

**Scrapeless** offers flexible and feature-rich data acquisition services with extensive parameter customization and multi-format export support. These capabilities empower LangChain to integrate and leverage external data more effectively. The core functional modules include:

**DeepSerp**
- **Google Search**: Enables comprehensive extraction of Google SERP data across all result types.
  - Supports selection of localized Google domains (e.g., `google.com`, `google.ad`) to retrieve region-specific search results.
  - Pagination supported for retrieving results beyond the first page.
  - Supports a search result filtering toggle to control whether to exclude duplicate or similar content.
- **Google Trends**: Retrieves keyword trend data from Google, including popularity over time, regional interest, and related searches.
  - Supports multi-keyword comparison.
  - Supports multiple data types: `interest_over_time`, `interest_by_region`, `related_queries`, and `related_topics`.
  - Allows filtering by specific Google properties (Web, YouTube, News, Shopping) for source-specific trend analysis.

**Universal Scraping**
- Designed for modern, JavaScript-heavy websites, allowing dynamic content extraction.
  - Global premium proxy support for bypassing geo-restrictions and improving reliability.

**Crawler**
- **Crawl**: Recursively crawl a website and its linked pages to extract site-wide content.
  - Supports configurable crawl depth and scoped URL targeting.
- **Scrape**: Extract content from a single webpage with high precision.
  - Supports "main content only" extraction to exclude ads, footers, and other non-essential elements.
  - Allows batch scraping of multiple standalone URLs.

## Overview

### Integration details

| Class | Package | Serializable | JS support |  Package latest |
| :--- | :--- | :---: | :---: | :---: |
| [ScrapelessUniversalScrapingTool](https://pypi.org/project/langchain-scrapeless/) | [langchain-scrapeless](https://pypi.org/project/langchain-scrapeless/) | ✅ | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapeless?style=flat-square&label=%20) |

### Tool features

|Native async|Returns artifact|Return data|
|:-:|:-:|:-:|
|✅|✅|html, markdown, links, metadata, structured content|


## Setup

The integration lives in the `langchain-scrapeless` package.

!pip install langchain-scrapeless

### Credentials

You'll need a Scrapeless API key to use this tool. You can set it as an environment variable:
"""
logger.info("# Scrapeless")


os.environ["SCRAPELESS_API_KEY"] = "your-api-key"

"""
## Instantiation

Here we show how to instantiate an instance of the Scrapeless Universal Scraping Tool. This tool allows you to scrape any website using a headless browser with JavaScript rendering capabilities, customizable output types, and geo-specific proxy support.

The tool accepts the following parameters during instantiation:
- `url` (required, str): The URL of the website to scrape.
- `headless` (optional, bool): Whether to use a headless browser. Default is True.
- `js_render` (optional, bool): Whether to enable JavaScript rendering. Default is True.
- `js_wait_until` (optional, str): Defines when to consider the JavaScript-rendered page ready. Default is `'domcontentloaded'`. Options include:
    - `load`: Wait until the page is fully loaded.
    - `domcontentloaded`: Wait until the DOM is fully loaded.
    - `networkidle0`: Wait until the network is idle.
    - `networkidle2`: Wait until the network is idle for 2 seconds.
- `outputs` (optional, str): The specific type of data to extract from the page. Options include:
    - `phone_numbers`
    - `headings`
    - `images`
    - `audios`
    - `videos`
    - `links`
    - `menus`
    - `hashtags`
    - `emails`
    - `metadata`
    - `tables`
    - `favicon`
- `response_type` (optional, str): Defines the format of the response. Default is `'html'`. Options include:
  - `html`: Return the raw HTML of the page.
  - `plaintext`: Return the plain text content.
  - `markdown`: Return a Markdown version of the page.
  - `png`: Return a PNG screenshot.
  - `jpeg`: Return a JPEG screenshot.
- `response_image_full_page` (optional, bool): Whether to capture and return a full-page image when using screenshot output (png or jpeg). Default is False.
- `selector` (optional, str): A specific CSS selector to scope scraping within a part of the page. Default is `None`.
- `proxy_country` (optional, str): Two-letter country code for geo-specific proxy access (e.g., `'us'`, `'gb'`, `'de'`, `'jp'`). Default is `'ANY'`.

## Invocation

### Basic Usage
"""
logger.info("## Instantiation")


tool = ScrapelessUniversalScrapingTool()

result = tool.invoke("https://example.com")
logger.debug(result)

"""
### Advanced Usage with Parameters
"""
logger.info("### Advanced Usage with Parameters")


tool = ScrapelessUniversalScrapingTool()

result = tool.invoke({"url": "https://exmaple.com", "response_type": "markdown"})
logger.debug(result)

"""
### Use within an agent
"""
logger.info("### Use within an agent")


llm = ChatOllama(model="llama3.2")

tool = ScrapelessUniversalScrapingTool()

tools = [tool]
agent = create_react_agent(llm, tools)

for chunk in agent.stream(
    {
        "messages": [
            (
                "human",
                "Use the scrapeless scraping tool to fetch https://www.scrapeless.com/en and extract the h1 tag.",
            )
        ]
    },
    stream_mode="values",
):
    chunk["messages"][-1].pretty_logger.debug()

"""
## API reference

- [Scrapeless Documentation](https://docs.scrapeless.com/en/universal-scraping-api/quickstart/introduction/)
- [Scrapeless API Reference](https://apidocs.scrapeless.com/api-12948840)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)