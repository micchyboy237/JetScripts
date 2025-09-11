from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_scrapeless import ScrapelessDeepSerpGoogleSearchTool
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
| [ScrapelessDeepSerpGoogleSearchTool](https://pypi.org/project/langchain-scrapeless/) | [langchain-scrapeless](https://pypi.org/project/langchain-scrapeless/) | ✅ | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapeless?style=flat-square&label=%20) |
| [ScrapelessDeepSerpGoogleTrendsTool](https://pypi.org/project/langchain-scrapeless/) | [langchain-scrapeless](https://pypi.org/project/langchain-scrapeless/) | ✅ | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapeless?style=flat-square&label=%20) |

### Tool features

|Native async|Returns artifact|Return data|
|:-:|:-:|:-:|
|✅|❌|Search Results Based on Tool|


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

### ScrapelessDeepSerpGoogleSearchTool

Here we show how to instantiate an instance of the `ScrapelessDeepSerpGoogleSearchTool`. The universal Information Search Engine allows you to retrieve any data information.
- Retrieves any data information.
- Handles explanatory queries (e.g., "why", "how").
- Supports comparative analysis requests.

The tool accepts the following parameters:
- `q`: (str) The search query string. Supports advanced Google syntax like `inurl:`, `site:`, `intitle:`, `as_eq`, etc.
- `hl`: (str) Language code for result content, e.g., `en`, `es`, `fr`. Default: `'en'`.
- `gl`: (str) Country code for geo-specific result targeting, e.g., `us`, `uk`, `de`. Default: `'us'`.
- `google_domain`: (str) Which Google domain to use (e.g., `'google.com'`, `'google.co.jp'`). Default: `'google.com'`.
- `start`: (int) Defines the result offset. It skips the given number of results. Used for pagination. Examples:
  - `0` (default): the first page of results
  - `10`: the second page
  - `20`: the third page
- `num`: (int) Defines the maximum number of results to return. Examples:
  - `10` (default): returns 10 results
  - `40`: returns 40 results
  - `100`: returns 100 results
- `ludocid`: (str) Defines the ID (CID) of the Google My Business listing you want to scrape. Also known as Google Place ID.
- `kgmid`: (str) Defines the ID (KGMID) of the Google Knowledge Graph listing you want to scrape. Also known as Google Knowledge Graph ID. Searches with the kgmid parameter will return results for the originally encrypted search parameters. For some searches, `kgmid` may override all other parameters except `start` and `num`.
- `ibp`: (str) Responsible for rendering layouts and expansions for some elements. Example: gwp;0,7 to expand searches with ludocid for expanded knowledge graph.
- `cr`: (str) Defines one or multiple countries to limit the search to. Uses format `country{two-letter country code}`, separated by `|`. Example:
  - `countryFR|countryDE` only searches French and German pages.
- `lr`: (str) Defines one or multiple languages to limit the search to. Uses format `lang_{two-letter language code}`, separated by `|`. Example:
  - `lang_fr|lang_de` only searches French and German pages.
- `tbs`: (str) Defines advanced search parameters not possible in the regular query field. Examples include advanced search for:
  - `patents`
  - `dates`
  - `news`
  - `videos`
  - `images`
  - `apps`
  - `text` contents
- `safe`: (str) Defines the level of filtering for adult content. Values:
  - `active`: blur explicit content
  - `off`: no filtering
- `nfpr`: (str) Defines exclusion of results from auto-corrected queries when the original query is misspelled. Values:
  - `1`: exclude these results
  - `0` (default): include them
  - Note: This may not prevent Google from returning auto-corrected results if no other results are available.
- `filter`: (str) Defines if `'Similar Results'` and `'Omitted Results'` filters are on or off. Values:
  - `1` (default): enable filters
  - `0`: disable filters
- `tbm`: (str) Defines the type of search to perform. Values:
  - `none`: regular Google Search
  - `isch`: Google Images
  - `lcl`: Google Local
  - `vid`: Google Videos
  - `nws`: Google News
  - `shop`: Google Shopping
  - `pts`: Google Patents
  - `jobs`: Google Jobs


### ScrapelessDeepSerpGoogleTrendsTool

Here we show how to instantiate an instance of the `ScrapelessDeepSerpGoogleTrendsTool`. This tool allows you to query real-time or historical trend data from Google Trends with fine control over locale, category, and result type, using the Scrapeless API.

The tool accepts the following parameters:
- `q` (required, str): Parameter defines the query or queries you want to search. You can use anything that you would use in a regular Google Trends search. The maximum number of queries per search is **5**. (This only applies to `interest_over_time` and `compared_breakdown_by_region` data types.) Other types of data will only accept **1 query** per search.
- `data_type` (optional, str): The type of data to retrieve. Default is `'interest_over_time'`. Options include:
  - `autocomplete`
  - `interest_over_time`
  - `compared_breakdown_by_region`
  - `interest_by_subregion`
  - `related_queries`
  - `related_topics`
- `date` (optional, str): Defines the date range to fetch data for. Default is `'today 1-m'`. Supported formats:
  - Relative: `'now 1-H'`, `'now 7-d'`, `'today 12-m'`, `'today 5-y'`, `'all'`
  - Custom date ranges: `'2023-01-01 2023-12-31'`
  - With hours: `'2023-07-01T10 2023-07-03T22'`
- `hl` (optional, str): Language code to use in the search. Default is `'en'`. Examples:
    - `'es'` (Spanish)
    - `'fr'` (French)
- `tz` (optional, str): Time zone offset. Default is `'420'` (PST).
- `geo` (optional, str): Two-letter country code to define the geographic origin of the search. Examples include:
  - `'US'` (United States)
  - `'GB'` (United Kingdom)
  - `'JP'` (Japan)
  - Leave empty or `None` for worldwide search.
- `cat` (optional, `CategoryEnum`): Category ID to narrow down the search context. Default is `'all_categories'` (0). Categories can include:
  - `'0'` – All categories
  - Others like `'3'` – News, `'29'` – Sports, etc.

## Invocation

### ScrapelessDeepSerpGoogleSearchTool

#### Basic Usage
"""
logger.info("## Instantiation")


tool = ScrapelessDeepSerpGoogleSearchTool()

result = tool.invoke("I want to know Scrapeless")
logger.debug(result)

"""
#### Advanced Usage with Parameters
"""
logger.info("#### Advanced Usage with Parameters")


tool = ScrapelessDeepSerpGoogleSearchTool()

result = tool.invoke({"q": "Scrapeless", "hl": "en", "google_domain": "google.com"})
logger.debug(result)

"""
#### Use within an agent
"""
logger.info("#### Use within an agent")


llm = ChatOllama(model="llama3.2")

tool = ScrapelessDeepSerpGoogleSearchTool()

tools = [tool]
agent = create_react_agent(llm, tools)

for chunk in agent.stream(
    {"messages": [("human", "I want to what is Scrapeless")]}, stream_mode="values"
):
    chunk["messages"][-1].pretty_logger.debug()

"""
### ScrapelessDeepSerpGoogleTrendsTool

#### Basic Usage
"""
logger.info("### ScrapelessDeepSerpGoogleTrendsTool")


tool = ScrapelessDeepSerpGoogleTrendsTool()

result = tool.invoke("Funny 2048, negamon monster trainer")
logger.debug(result)

"""
#### Advanced Usage with Parameters
"""
logger.info("#### Advanced Usage with Parameters")


tool = ScrapelessDeepSerpGoogleTrendsTool()

result = tool.invoke({"q": "Scrapeless", "data_type": "related_topics", "hl": "en"})
logger.debug(result)

"""
#### Use within an agent
"""
logger.info("#### Use within an agent")


llm = ChatOllama(model="llama3.2")

tool = ScrapelessDeepSerpGoogleTrendsTool()

tools = [tool]
agent = create_react_agent(llm, tools)

for chunk in agent.stream(
    {"messages": [("human", "I want to know the iPhone keyword trends")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_logger.debug()

"""
## API reference

- [Scrapeless Documentation](https://docs.scrapeless.com/en/deep-serp-api/quickstart/introduction/)
- [Scrapeless API Reference](https://apidocs.scrapeless.com/doc-800321)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)