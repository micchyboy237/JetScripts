from jet.logger import logger
from langchain_community.document_loaders import ScrapflyLoader
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
[ScrapFly](https://scrapfly.io/) is a web scraping API with headless browser capabilities, proxies, and anti-bot bypass. It allows for extracting web page data into accessible LLM markdown or text.

#### Installation
Install ScrapFly Python SDK and he required Langchain packages using pip:
```shell
pip install scrapfly-sdk langchain langchain-community
```

#### Usage
"""
logger.info("#### Installation")


scrapfly_loader = ScrapflyLoader(
    ["https://web-scraping.dev/products"],  # Get your API key from https://www.scrapfly.io/
    continue_on_failure=True,  # Ignore unprocessable web pages and log their exceptions
)

documents = scrapfly_loader.load()
logger.debug(documents)

"""
The ScrapflyLoader also allows passing ScrapeConfig object for customizing the scrape request. See the documentation for the full feature details and their API params: https://scrapfly.io/docs/scrape-api/getting-started
"""
logger.info("The ScrapflyLoader also allows passing ScrapeConfig object for customizing the scrape request. See the documentation for the full feature details and their API params: https://scrapfly.io/docs/scrape-api/getting-started")


scrapfly_scrape_config = {
    "asp": True,  # Bypass scraping blocking and antibot solutions, like Cloudflare
    "render_js": True,  # Enable JavaScript rendering with a cloud headless browser
    "proxy_pool": "public_residential_pool",  # Select a proxy pool (datacenter or residnetial)
    "country": "us",  # Select a proxy location
    "auto_scroll": True,  # Auto scroll the page
    "js": "",  # Execute custom JavaScript code by the headless browser
}

scrapfly_loader = ScrapflyLoader(
    ["https://web-scraping.dev/products"],  # Get your API key from https://www.scrapfly.io/
    continue_on_failure=True,  # Ignore unprocessable web pages and log their exceptions
    scrape_config=scrapfly_scrape_config,  # Pass the scrape_config object
    scrape_format="markdown",  # The scrape result format, either `markdown`(default) or `text`
)

documents = scrapfly_loader.load()
logger.debug(documents)

logger.info("\n\n[DONE]", bright=True)