from jet.logger import logger
from langchain_community.document_loaders import ScrapingAntLoader
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
sidebar_label: ScrapingAnt
---

# ScrapingAnt
[ScrapingAnt](https://scrapingant.com/) is a web scraping API with headless browser capabilities, proxies, and anti-bot bypass. It allows for extracting web page data into accessible LLM markdown.

This particular integration uses only Markdown extraction feature, but don't hesitate to [reach out to us](mailto:support@scrapingant.com) if you need more features provided by ScrapingAnt, but not yet implemented in this integration.

### Integration details

| Class                                                                                                                                                    | Package                                                                                        | Local | Serializable | JS support |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----:|:------------:|:----------:|
| [ScrapingAntLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.scrapingant.ScrapingAntLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) |   ❌   |      ❌       |     ❌      | 

### Loader features
|      Source       | Document Lazy Loading | Async Support |
|:-----------------:|:---------------------:|:-------------:| 
| ScrapingAntLoader |           ✅           |       ❌       |

## Setup

Install ScrapingAnt Python SDK and he required Langchain packages using pip:
```shell
pip install scrapingant-client langchain langchain-community
```

## Instantiation
"""
logger.info("# ScrapingAnt")


scrapingant_loader = ScrapingAntLoader(
    ["https://scrapingant.com/", "https://example.com/"],  # List of URLs to scrape
    # Get your API key from https://scrapingant.com/
    continue_on_failure=True,  # Ignore unprocessable web pages and log their exceptions
)

"""
The ScrapingAntLoader also allows providing a dict - scraping config for customizing the scrape request. As it is based on the [ScrapingAnt Python SDK](https://github.com/ScrapingAnt/scrapingant-client-python) you can pass any of the [common arguments](https://github.com/ScrapingAnt/scrapingant-client-python) to the `scrape_config` parameter.
"""
logger.info("The ScrapingAntLoader also allows providing a dict - scraping config for customizing the scrape request. As it is based on the [ScrapingAnt Python SDK](https://github.com/ScrapingAnt/scrapingant-client-python) you can pass any of the [common arguments](https://github.com/ScrapingAnt/scrapingant-client-python) to the `scrape_config` parameter.")


scrapingant_config = {
    "browser": True,  # Enable browser rendering with a cloud browser
    "proxy_type": "datacenter",  # Select a proxy type (datacenter or residential)
    "proxy_country": "us",  # Select a proxy location
}

scrapingant_additional_config_loader = ScrapingAntLoader(
    ["https://scrapingant.com/"],  # Get your API key from https://scrapingant.com/
    continue_on_failure=True,  # Ignore unprocessable web pages and log their exceptions
    scrape_config=scrapingant_config,  # Pass the scrape_config object
)

"""
## Load

Use the `load` method to scrape the web pages and get the extracted markdown content.
"""
logger.info("## Load")

documents = scrapingant_loader.load()

logger.debug(documents)

"""
## Lazy Load

Use the 'lazy_load' method to scrape the web pages and get the extracted markdown content lazily.
"""
logger.info("## Lazy Load")

lazy_documents = scrapingant_loader.lazy_load()

for document in lazy_documents:
    logger.debug(document)

"""
## API reference

This loader is based on the [ScrapingAnt Python SDK](https://docs.scrapingant.com/python-client). For more configuration options, see the [common arguments](https://github.com/ScrapingAnt/scrapingant-client-python/tree/master?tab=readme-ov-file#common-arguments)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)