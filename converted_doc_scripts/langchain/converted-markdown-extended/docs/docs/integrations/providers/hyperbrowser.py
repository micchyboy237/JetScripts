from jet.logger import logger
from langchain_hyperbrowser import HyperbrowserBrowserUseTool
from langchain_hyperbrowser import HyperbrowserClaudeComputerUseTool
from langchain_hyperbrowser import HyperbrowserCrawlTool
from langchain_hyperbrowser import HyperbrowserExtractTool
from langchain_hyperbrowser import HyperbrowserLoader
from langchain_hyperbrowser import HyperbrowserOllamaCUATool
from langchain_hyperbrowser import HyperbrowserScrapeTool
from pydantic import BaseModel
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
# Hyperbrowser

> [Hyperbrowser](https://hyperbrowser.ai) is a platform for running and scaling headless browsers. It lets you launch and manage browser sessions at scale and provides easy to use solutions for any webscraping needs, such as scraping a single page or crawling an entire site.
>
> Key Features:
>
> - Instant Scalability - Spin up hundreds of browser sessions in seconds without infrastructure headaches
> - Simple Integration - Works seamlessly with popular tools like Puppeteer and Playwright
> - Powerful APIs - Easy to use APIs for scraping/crawling any site, and much more
> - Bypass Anti-Bot Measures - Built-in stealth mode, ad blocking, automatic CAPTCHA solving, and rotating proxies

For more information about Hyperbrowser, please visit the [Hyperbrowser website](https://hyperbrowser.ai) or if you want to check out the docs, you can visit the [Hyperbrowser docs](https://docs.hyperbrowser.ai).

## Installation and Setup

To get started with `langchain-hyperbrowser`, you can install the package using pip:
"""
logger.info("# Hyperbrowser")

pip install langchain-hyperbrowser

"""
And you should configure credentials by setting the following environment variables:

`HYPERBROWSER_API_KEY=<your-api-key>`

Make sure to get your API Key from https://app.hyperbrowser.ai/

## Available Tools

Hyperbrowser provides two main categories of tools that are particularly useful for:
- Web scraping and data extraction from complex websites
- Automating repetitive web tasks
- Interacting with web applications that require authentication
- Performing research across multiple websites
- Testing web applications

### Browser Agent Tools

Hyperbrowser provides a number of Browser Agents tools. Currently we supported
 - Claude Computer Use
 - Ollama CUA
 - Browser Use

You can see more details [here](/docs/integrations/tools/hyperbrowser_browser_agent_tools)

#### Browser Use Tool
A general-purpose browser automation tool that can handle various web tasks through natural language instructions.
"""
logger.info("## Available Tools")


tool = HyperbrowserBrowserUseTool()
result = tool.run({
    "task": "Go to npmjs.com, find the React package, and tell me when it was last updated"
})
logger.debug(result)

"""
#### Ollama CUA Tool
Leverages Ollama's Computer Use Agent capabilities for advanced web interactions and information gathering.
"""
logger.info("#### Ollama CUA Tool")


tool = HyperbrowserOllamaCUATool()
result = tool.run({
    "task": "Go to Hacker News and summarize the top 5 posts right now"
})
logger.debug(result)

"""
#### Claude Computer Use Tool
Utilizes Ollama's Claude for sophisticated web browsing and information processing tasks.
"""
logger.info("#### Claude Computer Use Tool")


tool = HyperbrowserClaudeComputerUseTool()
result = tool.run({
    "task": "Go to GitHub's trending repositories page, and list the top 3 posts there right now"
})
logger.debug(result)

"""
### Web Scraping Tools

Here is a brief description of the Web Scraping Tools available with Hyperbrowser. You can see more details [here](/docs/integrations/tools/hyperbrowser_web_scraping_tools)

#### Scrape Tool
The Scrape Tool allows you to extract content from a single webpage in markdown, HTML, or link format.
"""
logger.info("### Web Scraping Tools")


tool = HyperbrowserScrapeTool()
result = tool.run({
    "url": "https://example.com",
    "scrape_options": {"formats": ["markdown"]}
})
logger.debug(result)

"""
#### Crawl Tool
The Crawl Tool enables you to traverse entire websites, starting from a given URL, with configurable page limits.
"""
logger.info("#### Crawl Tool")


tool = HyperbrowserCrawlTool()
result = tool.run({
    "url": "https://example.com",
    "max_pages": 2,
    "scrape_options": {"formats": ["markdown"]}
})
logger.debug(result)

"""
#### Extract Tool
The Extract Tool uses AI to pull structured data from web pages based on predefined schemas, making it perfect for data extraction tasks.
"""
logger.info("#### Extract Tool")


class SimpleExtractionModel(BaseModel):
    title: str

tool = HyperbrowserExtractTool()
result = tool.run({
    "url": "https://example.com",
    "schema": SimpleExtractionModel
})
logger.debug(result)

"""
## Document Loader

The `HyperbrowserLoader` class in `langchain-hyperbrowser` can easily be used to load content from any single page or multiple pages as well as crawl an entire site.
The content can be loaded as markdown or html.
"""
logger.info("## Document Loader")


loader = HyperbrowserLoader(urls="https://example.com")
docs = loader.load()

logger.debug(docs[0])

"""
### Advanced Usage

You can specify the operation to be performed by the loader. The default operation is `scrape`. For `scrape`, you can provide a single URL or a list of URLs to be scraped. For `crawl`, you can only provide a single URL. The `crawl` operation will crawl the provided page and subpages and return a document for each page.
"""
logger.info("### Advanced Usage")

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
  params={"scrape_options": {"include_tags": ["h1", "h2", "p"]}}
)

"""
## Additional Resources

- [Hyperbrowser Docs](https://docs.hyperbrowser.ai/)
- [GitHub](https://github.com/hyperbrowserai/langchain-hyperbrowser/)
- [PyPi](https://pypi.org/project/langchain-hyperbrowser/)
"""
logger.info("## Additional Resources")

logger.info("\n\n[DONE]", bright=True)