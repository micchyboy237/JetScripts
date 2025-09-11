from jet.logger import logger
from langchain_bright_data import BrightDataSERP
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
# Bright Data

[Bright Data](https://brightdata.com) is a web data platform that provides tools for web scraping, SERP collection, and accessing geo-restricted content.

Bright Data allows developers to extract structured data from websites, perform search engine queries, and access content that might be otherwise blocked or geo-restricted. The platform is designed to help overcome common web scraping challenges including anti-bot systems, CAPTCHAs, and IP blocks.

## Installation and Setup
"""
logger.info("# Bright Data")

pip install langchain-brightdata

"""
You'll need to set up your Bright Data API key:
"""
logger.info("You'll need to set up your Bright Data API key:")

os.environ["BRIGHT_DATA_API_KEY"] = "your-api-key"

"""
Or you can pass it directly when initializing tools:
"""
logger.info("Or you can pass it directly when initializing tools:")


tool = BrightDataSERP(bright_data_)

"""
## Tools

The Bright Data integration provides several tools:

- [BrightDataSERP](/docs/integrations/tools/brightdata_serp) - Search engine results collection with geo-targeting
- [BrightDataUnblocker](/docs/integrations/tools/brightdata_unlocker) - Access ANY public website that might be geo-restricted or bot-protected
- [BrightDataWebScraperAPI](/docs/integrations/tools/brightdata-webscraperapi) - Extract structured data from 100+ ppoular domains, e.g. Amazon product details and LinkedIn profiles
"""
logger.info("## Tools")

logger.info("\n\n[DONE]", bright=True)