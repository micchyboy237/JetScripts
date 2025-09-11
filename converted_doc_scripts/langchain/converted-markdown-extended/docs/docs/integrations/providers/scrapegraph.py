from jet.logger import logger
from langchain_scrapegraph.tools import (
SmartScraperTool,    # Extract structured data from websites
SmartCrawlerTool,    # Extract data from multiple pages with crawling
MarkdownifyTool,     # Convert webpages to markdown
GetCreditsTool,      # Check remaining API credits
)
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
# ScrapeGraph AI

>[ScrapeGraph AI](https://scrapegraphai.com) is a service that provides AI-powered web scraping capabilities.
>It offers tools for extracting structured data, converting webpages to markdown, and processing local HTML content
>using natural language prompts.

## Installation and Setup

Install the required packages:
"""
logger.info("# ScrapeGraph AI")

pip install langchain-scrapegraph

"""
Set up your API key:
"""
logger.info("Set up your API key:")

export SGAI_API_KEY="your-scrapegraph-api-key"

"""
## Tools

See a [usage example](/docs/integrations/tools/scrapegraph).

There are four tools available:
"""
logger.info("## Tools")


"""
Each tool serves a specific purpose:

- `SmartScraperTool`: Extract structured data from websites given a URL, prompt and optional output schema
- `SmartCrawlerTool`: Extract data from multiple pages with advanced crawling options like depth control, page limits, and domain restrictions
- `MarkdownifyTool`: Convert any webpage to clean markdown format
- `GetCreditsTool`: Check your remaining ScrapeGraph AI credits
"""
logger.info("Each tool serves a specific purpose:")

logger.info("\n\n[DONE]", bright=True)