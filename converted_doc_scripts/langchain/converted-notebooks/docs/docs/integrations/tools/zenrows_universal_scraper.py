from jet.adapters.langchain.chat_ollama import ChatOllama  # or your preferred LLM
from jet.logger import logger
from langchain_zenrows import ZenRowsUniversalScraper
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
# ZenRowsUniversalScraper

[ZenRows](https://www.zenrows.com/) is an enterprise-grade web scraping tool that provides advanced web data extraction capabilities at scale. For more information about ZenRows and its Universal Scraper API, visit the [official documentation](https://docs.zenrows.com/universal-scraper-api/).

This document provides a quick overview for getting started with ZenRowsUniversalScraper tool. For detailed documentation of all ZenRowsUniversalScraper features and configurations head to the [API reference](https://github.com/ZenRows-Hub/langchain-zenrows?tab=readme-ov-file#api-reference).

## Overview

### Integration details

| Class | Package | JS support |  Package latest |
| :--- | :--- | :---: | :---: |
| [ZenRowsUniversalScraper](https://pypi.org/project/langchain-zenrows/) | [langchain-zenrows](https://pypi.org/project/langchain-zenrows/) | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-zenrows?style=flat-square&label=%20) |

### Tool features

| Feature | Support |
| :--- | :---: |
| **JavaScript Rendering** | ✅ |
| **Anti-Bot Bypass** | ✅ |
| **Geo-Targeting** | ✅ |
| **Multiple Output Formats** | ✅ |
| **CSS Extraction** | ✅ |
| **Screenshot Capture** | ✅ |
| **Session Management** | ✅ |
| **Premium Proxies** | ✅ |

## Setup

Install the required dependencies.
"""
logger.info("# ZenRowsUniversalScraper")

pip install langchain-zenrows

"""
### Credentials

You'll need a ZenRows API key to use this tool. You can sign up for free at [ZenRows](https://app.zenrows.com/register?prod=universal_scraper).
"""
logger.info("### Credentials")


os.environ["ZENROWS_API_KEY"] = "<YOUR_ZENROWS_API_KEY>"

"""
## Instantiation

Here's how to instantiate an instance of the ZenRowsUniversalScraper tool.
"""
logger.info("## Instantiation")



os.environ["ZENROWS_API_KEY"] = "<YOUR_ZENROWS_API_KEY>"

zenrows_scraper_tool = ZenRowsUniversalScraper()

"""
You can also pass the ZenRows API key when initializing the ZenRowsUniversalScraper tool.
"""
logger.info("You can also pass the ZenRows API key when initializing the ZenRowsUniversalScraper tool.")


zenrows_scraper_tool = ZenRowsUniversalScraper(zenrows_)

"""
## Invocation

### Basic Usage

The tool accepts a URL and various optional parameters to customize the scraping behavior:
"""
logger.info("## Invocation")



os.environ["ZENROWS_API_KEY"] = "<YOUR_ZENROWS_API_KEY>"

zenrows_scraper_tool = ZenRowsUniversalScraper()

result = zenrows_scraper_tool.invoke({"url": "https://httpbin.io/html"})
logger.debug(result)

"""
### Advanced Usage with Parameters
"""
logger.info("### Advanced Usage with Parameters")



os.environ["ZENROWS_API_KEY"] = "<YOUR_ZENROWS_API_KEY>"

zenrows_scraper_tool = ZenRowsUniversalScraper()

result = zenrows_scraper_tool.invoke(
    {
        "url": "https://www.scrapingcourse.com/ecommerce/",
        "js_render": True,
        "premium_proxy": True,
        "proxy_country": "us",
        "response_type": "markdown",
        "wait": 2000,
    }
)

logger.debug(result)

"""
### Use within an agent
"""
logger.info("### Use within an agent")



os.environ["ZENROWS_API_KEY"] = "<YOUR_ZENROWS_API_KEY>"
# os.environ["OPENAI_API_KEY"] = "<YOUR_OPEN_AI_API_KEY>"


llm = ChatOllama(model="llama3.2")
zenrows_scraper_tool = ZenRowsUniversalScraper()

agent = create_react_agent(llm, [zenrows_scraper_tool])

result = agent.invoke(
    {
        "messages": "Scrape https://news.ycombinator.com/ and list the top 3 stories with title, points, comments, username, and time."
    }
)

logger.debug("Agent Response:")
for message in result["messages"]:
    logger.debug(f"{message.content}")

"""
## API reference

For detailed documentation of all ZenRowsUniversalScraper features and configurations head to the [**ZenRowsUniversalScraper API reference**](https://github.com/ZenRows-Hub/langchain-zenrows).

For comprehensive information about the underlying API parameters and capabilities, see the [ZenRows Universal API documentation](https://docs.zenrows.com/universal-scraper-api/api-reference).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)