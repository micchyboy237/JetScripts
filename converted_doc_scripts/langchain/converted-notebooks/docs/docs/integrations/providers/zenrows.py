from jet.logger import logger
from langchain_zenrows import ZenRowsUniversalScraper
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
# ZenRows

[ZenRows](https://www.zenrows.com/) is an enterprise-grade web scraping tool that provides advanced web data extraction capabilities at scale. ZenRows specializes in scraping modern websites, bypassing anti-bot systems, extracting structured data from any website, rendering JavaScript-heavy content, accessing geo-restricted websites, and more.

[langchain-zenrows](https://pypi.org/project/langchain-zenrows/) provides tools that allow LLMs to access web data using ZenRows' powerful scraping infrastructure.

## Installation and Setup

```bash
pip install langchain-zenrows
```

You'll need to set up your ZenRows API key:

```python
os.environ["ZENROWS_API_KEY"] = "your-api-key"
```

Or you can pass it directly when initializing tools:

```python
zenrows_scraper_tool = ZenRowsUniversalScraper(zenrows_)
```

## Tools

### ZenRowsUniversalScraper

The ZenRows integration provides comprehensive web scraping features:

- **JavaScript Rendering**: Scrape modern SPAs and dynamic content
- **Anti-Bot Bypass**: Overcome sophisticated bot detection systems  
- **Geo-Targeting**: Access region-specific content with 190+ countries
- **Multiple Output Formats**: HTML, Markdown, Plaintext, PDF, Screenshots
- **CSS Extraction**: Target specific data with CSS selectors
- **Structured Data Extraction**: Automatically extract emails, phone numbers, links, and more
- **Session Management**: Maintain consistent sessions across requests
- **Premium Proxies**: Residential IPs for maximum success rates

See more in the [ZenRows tool documentation](/docs/integrations/tools/zenrows_universal_scraper).
"""
logger.info("# ZenRows")

logger.info("\n\n[DONE]", bright=True)