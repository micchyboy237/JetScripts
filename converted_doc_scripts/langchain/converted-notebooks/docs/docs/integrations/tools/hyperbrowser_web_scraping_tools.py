from jet.transformers.formatters import format_json
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_hyperbrowser import (
HyperbrowserCrawlTool,
HyperbrowserExtractTool,
HyperbrowserScrapeTool,
)
from langchain_hyperbrowser import HyperbrowserCrawlTool
from langchain_hyperbrowser import HyperbrowserExtractTool
from langchain_hyperbrowser import HyperbrowserScrapeTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing import List
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
---
sidebar_label: Hyperbrowser Web Scraping Tools
---

# Hyperbrowser Web Scraping Tools

[Hyperbrowser](https://hyperbrowser.ai) is a platform for running and scaling headless browsers. It lets you launch and manage browser sessions at scale and provides easy to use solutions for any webscraping needs, such as scraping a single page or crawling an entire site.

Key Features:

- Instant Scalability - Spin up hundreds of browser sessions in seconds without infrastructure headaches
- Simple Integration - Works seamlessly with popular tools like Puppeteer and Playwright
- Powerful APIs - Easy to use APIs for scraping/crawling any site, and much more
- Bypass Anti-Bot Measures - Built-in stealth mode, ad blocking, automatic CAPTCHA solving, and rotating proxies

This notebook provides a quick overview for getting started with Hyperbrowser web tools.

For more information about Hyperbrowser, please visit the [Hyperbrowser website](https://hyperbrowser.ai) or if you want to check out the docs, you can visit the [Hyperbrowser docs](https://docs.hyperbrowser.ai).

## Key Capabilities

### Scrape

Hyperbrowser provides powerful scraping capabilities that allow you to extract data from any webpage. The scraping tool can convert web content into structured formats like markdown or HTML, making it easy to process and analyze the data.

### Crawl

The crawling functionality enables you to navigate through multiple pages of a website automatically. You can set parameters like page limits to control how extensively the crawler explores the site, collecting data from each page it visits.

### Extract

Hyperbrowser's extraction capabilities use AI to pull specific information from webpages according to your defined schema. This allows you to transform unstructured web content into structured data that matches your exact requirements.

## Overview

### Integration details

| Tool         | Package                | Local | Serializable | JS support |
| :----------- | :--------------------- | :---: | :----------: | :--------: |
| Crawl Tool   | langchain-hyperbrowser |  ❌   |      ❌      |     ❌     |
| Scrape Tool  | langchain-hyperbrowser |  ❌   |      ❌      |     ❌     |
| Extract Tool | langchain-hyperbrowser |  ❌   |      ❌      |     ❌     |

## Setup

To access the Hyperbrowser web tools you'll need to install the `langchain-hyperbrowser` integration package, and create a Hyperbrowser account and get an API key.

### Credentials

Head to [Hyperbrowser](https://app.hyperbrowser.ai/) to sign up and generate an API key. Once you've done this set the HYPERBROWSER_API_KEY environment variable:

```bash
export HYPERBROWSER_API_KEY=<your-api-key>
```

### Installation

Install **langchain-hyperbrowser**.
"""
logger.info("# Hyperbrowser Web Scraping Tools")

# %pip install -qU langchain-hyperbrowser

"""
## Instantiation

### Crawl Tool

The `HyperbrowserCrawlTool` is a powerful tool that can crawl entire websites, starting from a given URL. It supports configurable page limits and scraping options.

```python
tool = HyperbrowserCrawlTool()
```

### Scrape Tool

The `HyperbrowserScrapeTool` is a tool that can scrape content from web pages. It supports both markdown and HTML output formats, along with metadata extraction.

```python
tool = HyperbrowserScrapeTool()
```

### Extract Tool

The `HyperbrowserExtractTool` is a powerful tool that uses AI to extract structured data from web pages. It can extract information based predefined schemas.

```python
tool = HyperbrowserExtractTool()
```

## Invocation

### Basic Usage

#### Crawl Tool
"""
logger.info("## Instantiation")


result = HyperbrowserCrawlTool().invoke(
    {
        "url": "https://example.com",
        "max_pages": 2,
        "scrape_options": {"formats": ["markdown"]},
    }
)
logger.debug(result)

"""
#### Scrape Tool
"""
logger.info("#### Scrape Tool")


result = HyperbrowserScrapeTool().invoke(
    {"url": "https://example.com", "scrape_options": {"formats": ["markdown"]}}
)
logger.debug(result)

"""
#### Extract Tool
"""
logger.info("#### Extract Tool")



class SimpleExtractionModel(BaseModel):
    title: str


result = HyperbrowserExtractTool().invoke(
    {
        "url": "https://example.com",
        "schema": SimpleExtractionModel,
    }
)
logger.debug(result)

"""
### With Custom Options

#### Crawl Tool with Custom Options
"""
logger.info("### With Custom Options")

result = HyperbrowserCrawlTool().run(
    {
        "url": "https://example.com",
        "max_pages": 2,
        "scrape_options": {
            "formats": ["markdown", "html"],
        },
        "session_options": {"use_proxy": True, "solve_captchas": True},
    }
)
logger.debug(result)

"""
#### Scrape Tool with Custom Options
"""
logger.info("#### Scrape Tool with Custom Options")

result = HyperbrowserScrapeTool().run(
    {
        "url": "https://example.com",
        "scrape_options": {
            "formats": ["markdown", "html"],
        },
        "session_options": {"use_proxy": True, "solve_captchas": True},
    }
)
logger.debug(result)

"""
#### Extract Tool with Custom Schema
"""
logger.info("#### Extract Tool with Custom Schema")




class ProductSchema(BaseModel):
    title: str
    price: float


class ProductsSchema(BaseModel):
    products: List[ProductSchema]


result = HyperbrowserExtractTool().run(
    {
        "url": "https://dummyjson.com/products?limit=10",
        "schema": ProductsSchema,
        "session_options": {"session_options": {"use_proxy": True}},
    }
)
logger.debug(result)

"""
### Async Usage

All tools support async usage:
"""
logger.info("### Async Usage")




class ExtractionSchema(BaseModel):
    popular_library_name: List[str]


async def web_operations():
    crawl_tool = HyperbrowserCrawlTool()
    crawl_result = await crawl_tool.arun(
            {
                "url": "https://example.com",
                "max_pages": 5,
                "scrape_options": {"formats": ["markdown"]},
            }
        )
    logger.success(format_json(crawl_result))

    scrape_tool = HyperbrowserScrapeTool()
    scrape_result = await scrape_tool.arun(
            {"url": "https://example.com", "scrape_options": {"formats": ["markdown"]}}
        )
    logger.success(format_json(scrape_result))

    extract_tool = HyperbrowserExtractTool()
    extract_result = await extract_tool.arun(
            {
                "url": "https://npmjs.com",
                "schema": ExtractionSchema,
            }
        )
    logger.success(format_json(extract_result))

    return crawl_result, scrape_result, extract_result


results = await web_operations()
logger.success(format_json(results))
logger.debug(results)

"""
## Use within an agent

Here's how to use any of the web tools within an agent:
"""
logger.info("## Use within an agent")


crawl_tool = HyperbrowserCrawlTool()

llm = ChatOllama(model="llama3.2")

agent = create_react_agent(llm, [crawl_tool])
user_input = "Crawl https://example.com and get content from up to 5 pages"
for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## Configuration Options

### Common Options

All tools support these basic configuration options:

- `url`: The URL to process
- `session_options`: Browser session configuration
  - `use_proxy`: Whether to use a proxy
  - `solve_captchas`: Whether to automatically solve CAPTCHAs
  - `accept_cookies`: Whether to accept cookies

### Tool-Specific Options

#### Crawl Tool

- `max_pages`: Maximum number of pages to crawl
- `scrape_options`: Options for scraping each page
  - `formats`: List of output formats (markdown, html)

#### Scrape Tool

- `scrape_options`: Options for scraping the page
  - `formats`: List of output formats (markdown, html)

#### Extract Tool

- `schema`: Pydantic model defining the structure to extract
- `extraction_prompt`: Natural language prompt for extraction

For more details, see the respective API references:

- [Crawl API Reference](https://docs.hyperbrowser.ai/reference/api-reference/crawl)
- [Scrape API Reference](https://docs.hyperbrowser.ai/reference/api-reference/scrape)
- [Extract API Reference](https://docs.hyperbrowser.ai/reference/api-reference/extract)

## API reference

- [GitHub](https://github.com/hyperbrowserai/langchain-hyperbrowser/)
- [PyPi](https://pypi.org/project/langchain-hyperbrowser/)
- [Hyperbrowser Docs](https://docs.hyperbrowser.ai/)
"""
logger.info("## Configuration Options")

logger.info("\n\n[DONE]", bright=True)