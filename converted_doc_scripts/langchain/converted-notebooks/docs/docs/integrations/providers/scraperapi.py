from jet.logger import logger
from langchain_scraperapi.tools import (
ScraperAPIAmazonSearchTool,
ScraperAPIGoogleSearchTool,
ScraperAPITool,
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
# ScraperAPI

[ScraperAPI](https://www.scraperapi.com/) enables data collection from any public website with its web scraping API, without worrying about proxies, browsers, or CAPTCHA handling. [langchain-scraperapi](https://github.com/scraperapi/langchain-scraperapi) wraps this service, making it easy for AI agents to browse the web and scrape data from it.

## Installation and Setup

- Install the Python package with `pip install langchain-scraperapi`.
- Obtain an API key from [ScraperAPI](https://www.scraperapi.com/) and set the environment variable `SCRAPERAPI_API_KEY`.

### Tools

The package offers 3 tools to scrape any website, get structured Google search results, and get structured Amazon search results respectively.

To import them:
"""
logger.info("# ScraperAPI")

# %pip install langchain_scraperapi


"""
Example use:
"""
logger.info("Example use:")

tool = ScraperAPITool()

result = tool.invoke({"url": "https://example.com", "output_format": "markdown"})
logger.debug(result)

"""
For a more detailed walkthrough of how to use these tools, visit the [official repository](https://github.com/scraperapi/langchain-scraperapi).
"""
logger.info("For a more detailed walkthrough of how to use these tools, visit the [official repository](https://github.com/scraperapi/langchain-scraperapi).")

logger.info("\n\n[DONE]", bright=True)