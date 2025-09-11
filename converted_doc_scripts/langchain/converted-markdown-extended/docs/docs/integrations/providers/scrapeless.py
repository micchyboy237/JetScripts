from jet.logger import logger
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

[Scrapeless](https://scrapeless.com) offers flexible and feature-rich data acquisition services with extensive parameter customization and multi-format export support.

## Installation and Setup
"""
logger.info("# Scrapeless")

pip install langchain-scrapeless

"""
You'll need to set up your Scrapeless API key:
"""
logger.info("You'll need to set up your Scrapeless API key:")

os.environ["SCRAPELESS_API_KEY"] = "your-api-key"

"""
## Tools

The Scrapeless integration provides several tools:

- [ScrapelessDeepSerpGoogleSearchTool](/docs/integrations/tools/scrapeless_scraping_api) - Enables comprehensive extraction of Google SERP data across all result types.
- [ScrapelessDeepSerpGoogleTrendsTool](/docs/integrations/tools/scrapeless_scraping_api) - Retrieves keyword trend data from Google, including popularity over time, regional interest, and related searches.
- [ScrapelessUniversalScrapingTool](/docs/integrations/tools/scrapeless_universal_scraping) - Access and extract data from JS-Render websites that typically block bots.
- [ScrapelessCrawlerCrawlTool](/docs/integrations/tools/scrapeless_crawl) - Crawl a website and its linked pages to extract comprehensive data.
- [ScrapelessCrawlerScrapeTool](/docs/integrations/tools/scrapeless_crawl) - Extract information from a single webpage.
"""
logger.info("## Tools")

logger.info("\n\n[DONE]", bright=True)