from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_scrapeless import ScrapelessCrawlerCrawlTool
from langchain_scrapeless import ScrapelessCrawlerScrapeTool
from langchain_scrapeless import ScrapelessDeepSerpGoogleTrendsTool
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

[**Scrapeless**](https://www.scrapeless.com/) offers flexible and feature-rich data acquisition services with extensive parameter customization and multi-format export support. These capabilities empower LangChain to integrate and leverage external data more effectively. The core functional modules include:

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
| [ScrapelessCrawlerScrapeTool](https://pypi.org/project/langchain-scrapeless/) | [langchain-scrapeless](https://pypi.org/project/langchain-scrapeless/) | ✅ | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapeless?style=flat-square&label=%20) |
| [ScrapelessCrawlerCrawlTool](https://pypi.org/project/langchain-scrapeless/) | [langchain-scrapeless](https://pypi.org/project/langchain-scrapeless/) | ✅ | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapeless?style=flat-square&label=%20) |

### Tool features

|Native async|Returns artifact|Return data|
|:-:|:-:|:-:|
|✅|✅|markdown, rawHtml, screenshot@fullPage, json, links, screenshot, html|


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

### ScrapelessCrawlerScrapeTool

The ScrapelessCrawlerScrapeTool allows you to scrape content from one or multiple websites using Scrapeless’s Crawler Scrape API. You can extract the main content, control formatting, headers, wait times, and output types.


The tool accepts the following parameters:
- `urls` (required, List[str]): One or more URLs of websites you want to scrape.
- `formats` (optional, List[str]): Defines the format(s) of the scraped output. Default is `['markdown']`. Options include:
  - `'markdown'`
  - `'rawHtml'`
  - `'screenshot@fullPage'`
  - `'json'`
  - `'links'`
  - `'screenshot'`
  - `'html'`
- `only_main_content` (optional, bool): Whether to return only the main page content, excluding headers, navs, footers, etc. Default is True.
- `include_tags` (optional, List[str]): A list of HTML tags to include in the output (e.g., `['h1', 'p']`). If set to None, no tags are explicitly included.
- `exclude_tags` (optional, List[str]): A list of HTML tags to exclude from the output. If set to None, no tags are explicitly excluded.
- `headers` (optional, Dict[str, str]): Custom headers to send with the request (e.g., for cookies or user-agent). Default is None.
- `wait_for` (optional, int): Time to wait in milliseconds before scraping. Useful for giving the page time to fully load. Default is `0`.
- `timeout` (optional, int): Request timeout in milliseconds. Default is `30000`.

### ScrapelessCrawlerCrawlTool

The ScrapelessCrawlerCrawlTool allows you to crawl a website starting from a base URL using Scrapeless’s Crawler Crawl API. It supports advanced filtering of URLs, crawl depth control, content scraping options, headers customization, and more.

The tool accepts the following parameters:
- `url` (required, str): The base URL to start crawling from.

- `limit` (optional, int): Maximum number of pages to crawl. Default is `10000`.
- `include_paths` (optional, List[str]): URL pathname regex patterns to include matching URLs in the crawl. Only URLs matching these patterns will be included. For example, setting `["blog/.*"]` will only include URLs under the `/blog/` path. Default is None.
- `exclude_paths` (optional, List[str]): URL pathname regex patterns to exclude matching URLs from the crawl. For example, setting `["blog/.*"]` will exclude URLs under the `/blog/` path. Default is None.
- `max_depth` (optional, int): Maximum crawl depth relative to the base URL, measured by the number of slashes in the URL path. Default is `10`.
- `max_discovery_depth` (optional, int): Maximum crawl depth based on discovery order. Root and sitemapped pages have depth `0`. For example, setting to `1` and ignoring sitemap will crawl only the entered URL and its immediate links. Default is None.
- `ignore_sitemap` (optional, bool): Whether to ignore the website sitemap during crawling. Default is False.
- `ignore_query_params` (optional, bool): Whether to ignore query parameter differences to avoid re-scraping similar URLs. Default is False.
- `deduplicate_similar_urls` (optional, bool): Whether to deduplicate similar URLs. Default is True.
- `regex_on_full_url` (optional, bool): Whether regex matching applies to the full URL instead of just the path. Default is True.
- `allow_backward_links` (optional, bool): Whether to allow crawling backlinks outside the URL hierarchy. Default is False.
- `allow_external_links` (optional, bool): Whether to allow crawling links to external websites. Default is False.
- `delay` (optional, int): Delay in seconds between page scrapes to respect rate limits. Default is `1`.
- `formats` (optional, List[str]): The format(s) of the scraped content. Default is ["markdown"]. Options include:
  - `'markdown'`
  - `'rawHtml'`
  - `'screenshot@fullPage'`
  - `'json'`
  - `'links'`
  - `'screenshot'`
  - `'html'`
- `only_main_content` (optional, bool): Whether to return only the main content, excluding headers, navigation bars, footers, etc. Default is True.
- `include_tags` (optional, List[str]): List of HTML tags to include in the output (e.g., `['h1', 'p']`). Default is None (no explicit include filter).
- `exclude_tags` (optional, List[str]): List of HTML tags to exclude from the output. Default is None (no explicit exclude filter).
- `headers` (optional, Dict[str, str]): Custom HTTP headers to send with the requests, such as cookies or user-agent strings. Default is None.
- `wait_for` (optional, int): Time in milliseconds to wait before scraping the content, allowing the page to load fully. Default is `0`.
- `timeout` (optional, int):Request timeout in milliseconds. Default is `30000`.

## Invocation

### ScrapelessCrawlerCrawlTool

#### Usage with Parameters
"""
logger.info("## Instantiation")


tool = ScrapelessCrawlerCrawlTool()

result = tool.invoke({"url": "https://exmaple.com", "limit": 4})
logger.debug(result)

"""
#### Use within an agent
"""
logger.info("#### Use within an agent")


llm = ChatOllama(model="llama3.2")

tool = ScrapelessCrawlerCrawlTool()

tools = [tool]
agent = create_react_agent(llm, tools)

for chunk in agent.stream(
    {
        "messages": [
            (
                "human",
                "Use the scrapeless crawler crawl tool to crawl the website https://example.com and output the markdown content as a string.",
            )
        ]
    },
    stream_mode="values",
):
    chunk["messages"][-1].pretty_logger.debug()

"""
### ScrapelessCrawlerScrapeTool

#### Usage with Parameters
"""
logger.info("### ScrapelessCrawlerScrapeTool")


tool = ScrapelessDeepSerpGoogleTrendsTool()

result = tool.invoke("Funny 2048,negamon monster trainer")
logger.debug(result)

"""
#### Advanced Usage with Parameters
"""
logger.info("#### Advanced Usage with Parameters")


tool = ScrapelessCrawlerScrapeTool()

result = tool.invoke(
    {
        "urls": ["https://exmaple.com", "https://www.scrapeless.com/en"],
        "formats": ["markdown"],
    }
)
logger.debug(result)

"""
#### Use within an agent
"""
logger.info("#### Use within an agent")


llm = ChatOllama(model="llama3.2")

tool = ScrapelessCrawlerScrapeTool()

tools = [tool]
agent = create_react_agent(llm, tools)

for chunk in agent.stream(
    {
        "messages": [
            (
                "human",
                "Use the scrapeless crawler scrape tool to get the website content of https://example.com and output the html content as a string.",
            )
        ]
    },
    stream_mode="values",
):
    chunk["messages"][-1].pretty_logger.debug()

"""
## API reference

- [Scrapeless Documentation](https://docs.scrapeless.com/en/crawl/quickstart/introduction/)
- [Scrapeless API Reference](https://apidocs.scrapeless.com/api-17509003)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)