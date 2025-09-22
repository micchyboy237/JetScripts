from IPython.display import Markdown, display
from jet.logger import logger
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import AgentQLWebReader
from llama_index.readers.web import BrowserbaseWebReader
from llama_index.readers.web import FireCrawlWebReader
from llama_index.readers.web import HyperbrowserWebReader
from llama_index.readers.web import OlostepWebReader
from llama_index.readers.web import OxylabsWebReader
from llama_index.readers.web import RssReader
from llama_index.readers.web import ScrapflyReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.web import SpiderWebReader
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.readers.web import ZenRowsWebReader
from llama_index.readers.web import ZyteWebReader
from llama_index.readers.web.firecrawl_web.base import FireCrawlWebReader
import logging
import os
import shutil
import sys


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/WebPageDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Web Page Reader

Demonstrates our web page reader.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Web Page Reader")

# %pip install llama-index llama-index-readers-web


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
#### Using SimpleWebPageReader
"""
logger.info("#### Using SimpleWebPageReader")




documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["http://paulgraham.com/worked.html"]
)

documents[0]

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

"""
# Using Spider Reader 🕷
[Spider](https://spider.cloud/?ref=llama_index) is the [fastest](https://github.com/spider-rs/spider/blob/main/benches/BENCHMARKS.md#benchmark-results) crawler. It converts any website into pure HTML, markdown, metadata or text while enabling you to crawl with custom actions using AI.

Spider allows you to use high performance proxies to prevent detection, caches AI actions, webhooks for crawling status, scheduled crawls etc... 

**Prerequisites:** you need to have a Spider api key to use this loader. You can get one on [spider.cloud](https://spider.cloud).
"""
logger.info("# Using Spider Reader 🕷")


spider_reader = SpiderWebReader(
    # Get one at https://spider.cloud
    mode="scrape",
)

documents = spider_reader.load_data(url="https://spider.cloud")
logger.debug(documents)

"""
Crawl domain following all deeper subpages
"""
logger.info("Crawl domain following all deeper subpages")


spider_reader = SpiderWebReader(
    mode="crawl",
)

documents = spider_reader.load_data(url="https://spider.cloud")
logger.debug(documents)

"""
For guides and documentation, visit [Spider](https://spider.cloud/docs/api)

# Using Browserbase Reader 🅱️

[Browserbase](https://browserbase.com) is a serverless platform for running headless browsers, it offers advanced debugging, session recordings, stealth mode, integrated proxies and captcha solving.

## Installation and Setup

- Get an API key and Project ID from [browserbase.com](https://browserbase.com) and set it in environment variables (`BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID`).
- Install the [Browserbase SDK](http://github.com/browserbase/python-sdk):
"""
logger.info("# Using Browserbase Reader 🅱️")

# %pip install browserbase


reader = BrowserbaseWebReader()
docs = reader.load_data(
    urls=[
        "https://example.com",
    ],
    text_content=False,
)

"""
### Using FireCrawl Reader 🔥

Firecrawl is an api that turns entire websites into clean, LLM accessible markdown.

Using Firecrawl to gather an entire website
"""
logger.info("### Using FireCrawl Reader 🔥")


firecrawl_reader = FireCrawlWebReader(
    # Replace with your actual API key from https://www.firecrawl.dev/
    mode="scrape",  # Choose between "crawl" and "scrape" for single page scraping
    params={"additional": "parameters"},  # Optional additional parameters
)

documents = firecrawl_reader.load_data(url="http://paulgraham.com/")

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

"""
Using firecrawl for a single page
"""
logger.info("Using firecrawl for a single page")


firecrawl_reader = FireCrawlWebReader(
    # Replace with your actual API key from https://www.firecrawl.dev/
    mode="scrape",  # Choose between "crawl" and "scrape" for single page scraping
    params={"additional": "parameters"},  # Optional additional parameters
)

documents = firecrawl_reader.load_data(url="http://paulgraham.com/worked.html")

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

"""
Using FireCrawl's extract mode to extract structured data from URLs
"""
logger.info("Using FireCrawl's extract mode to extract structured data from URLs")


firecrawl_reader = FireCrawlWebReader(
    # Replace with your actual API key from https://www.firecrawl.dev/
    mode="extract",  # Use extract mode to extract structured data
    params={
        "prompt": "Extract the title, author, and main points from this essay",
    },
)

documents = firecrawl_reader.load_data(
    urls=[
        "https://www.paulgraham.com",
        "https://www.paulgraham.com/worked.html",
    ]
)

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What are the main points from these essays?")

display(Markdown(f"<b>{response}</b>"))

"""
# Using Hyperbrowser Reader ⚡

[Hyperbrowser](https://hyperbrowser.ai) is a platform for running and scaling headless browsers. It lets you launch and manage browser sessions at scale and provides easy to use solutions for any webscraping needs, such as scraping a single page or crawling an entire site.

Key Features:
- Instant Scalability - Spin up hundreds of browser sessions in seconds without infrastructure headaches
- Simple Integration - Works seamlessly with popular tools like Puppeteer and Playwright
- Powerful APIs - Easy to use APIs for scraping/crawling any site, and much more
- Bypass Anti-Bot Measures - Built-in stealth mode, ad blocking, automatic CAPTCHA solving, and rotating proxies

For more information about Hyperbrowser, please visit the [Hyperbrowser website](https://hyperbrowser.ai) or if you want to check out the docs, you can visit the [Hyperbrowser docs](https://docs.hyperbrowser.ai).

## Installation and Setup

- Head to [Hyperbrowser](https://app.hyperbrowser.ai/) to sign up and generate an API key. Once you've done this set the `HYPERBROWSER_API_KEY` environment variable or you can pass it to the `HyperbrowserWebReader` constructor.
- Install the [Hyperbrowser SDK](https://github.com/hyperbrowserai/python-sdk):
"""
logger.info("# Using Hyperbrowser Reader ⚡")

# %pip install hyperbrowser


reader = HyperbrowserWebReader()
docs = reader.load_data(
    urls=["https://example.com"],
    operation="scrape",
)
docs

"""
#### Using TrafilaturaWebReader
"""
logger.info("#### Using TrafilaturaWebReader")


documents = TrafilaturaWebReader().load_data(
    ["http://paulgraham.com/worked.html"]
)

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

"""
### Using RssReader
"""
logger.info("### Using RssReader")


documents = RssReader().load_data(
    ["https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"]
)

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What happened in the news today?")

"""
## Using ScrapFly
ScrapFly is a web scraping API with headless browser capabilities, proxies, and anti-bot bypass. It allows for extracting web page data into accessible LLM markdown or text. Install ScrapFly Python SDK using pip:
```shell
pip install scrapfly-sdk
```

Here is a basic usage of ScrapflyReader
"""
logger.info("## Using ScrapFly")


scrapfly_reader = ScrapflyReader(
    # Get your API key from https://www.scrapfly.io/
    ignore_scrape_failures=True,  # Ignore unprocessable web pages and log their exceptions
)

documents = scrapfly_reader.load_data(
    urls=["https://web-scraping.dev/products"]
)

"""
The ScrapflyReader also allows passigng ScrapeConfig object for customizing the scrape request. See the documentation for the full feature details and their API params: https://scrapfly.io/docs/scrape-api/getting-started
"""
logger.info("The ScrapflyReader also allows passigng ScrapeConfig object for customizing the scrape request. See the documentation for the full feature details and their API params: https://scrapfly.io/docs/scrape-api/getting-started")


scrapfly_reader = ScrapflyReader(
    # Get your API key from https://www.scrapfly.io/
    ignore_scrape_failures=True,  # Ignore unprocessable web pages and log their exceptions
)

scrapfly_scrape_config = {
    "asp": True,  # Bypass scraping blocking and antibot solutions, like Cloudflare
    "render_js": True,  # Enable JavaScript rendering with a cloud headless browser
    "proxy_pool": "public_residential_pool",  # Select a proxy pool (datacenter or residnetial)
    "country": "us",  # Select a proxy location
    "auto_scroll": True,  # Auto scroll the page
    "js": "",  # Execute custom JavaScript code by the headless browser
}

documents = scrapfly_reader.load_data(
    urls=["https://web-scraping.dev/products"],
    scrape_config=scrapfly_scrape_config,  # Pass the scrape config
    scrape_format="markdown",  # The scrape result format, either `markdown`(default) or `text`
)

"""
# Using ZyteWebReader

ZyteWebReader allows a user to access the content of webpage in different modes ("article", "html-text", "html"). 
It enables user to change setting such as browser rendering and JS as the content of many sites would require setting these options to access relevant content. All supported options can be found here: https://docs.zyte.com/zyte-api/usage/reference.html

To install dependencies:
```shell
pip install zyte-api
```

To get access to your ZYTE API key please visit: https://docs.zyte.com/zyte-api/get-started.html
"""
logger.info("# Using ZyteWebReader")




zyte_reader = ZyteWebReader(
    mode="article",  # or "html-text" or "html"
)

urls = [
    "https://www.zyte.com/blog/web-scraping-apis/",
    "https://www.zyte.com/blog/system-integrators-extract-big-data/",
]

documents = zyte_reader.load_data(
    urls=urls,
)

logger.debug(len(documents[0].text))

"""
Browser rendering and javascript can be enabled by passing setting corresponding parameters during initialization.
"""
logger.info("Browser rendering and javascript can be enabled by passing setting corresponding parameters during initialization.")

zyte_dw_params = {
    "browserHtml": True,  # Enable browser rendering
    "javascript": True,  # Enable JavaScript
}

zyte_reader = ZyteWebReader(
    download_kwargs=zyte_dw_params,
)

documents = zyte_reader.load_data(
    urls=urls,
)

len(documents[0].text)

"""
Set "continue_on_failure" to False if you'd like to stop when any request fails.
"""
logger.info("Set "continue_on_failure" to False if you'd like to stop when any request fails.")

zyte_reader = ZyteWebReader(
    mode="html-text",
    download_kwargs=zyte_dw_params,
    continue_on_failure=False,
)

documents = zyte_reader.load_data(
    urls=urls,
)

len(documents[0].text)

"""
In default mode ("article") only the article text is extracted while in the "html-text" full text is extracted from the webpage, there the length of the text is significantly longer.

# Using AgentQLWebReader 🐠

Use AgentQL to scrape structured data from a website.
"""
logger.info("# Using AgentQLWebReader 🐠")


agentql_reader = AgentQLWebReader(
    # Replace with your actual API key from https://dev.agentql.com
    params={
        "is_scroll_to_bottom_enabled": True
    },  # Optional additional parameters
)

document = agentql_reader.load_data(
    url="https://www.ycombinator.com/companies?batch=W25",
    query="{ company[] { name location description industry_category link(a link to the company's detail on Ycombinator)} }",
)

index = VectorStoreIndex.from_documents(document)
query_engine = index.as_query_engine()
response = query_engine.query(
    "Find companies that are working on web agent, list their names, locations and link"
)

display(Markdown(f"<b>{response}</b>"))

"""
# Using OxylabsWebReader

OxylabsWebReader allows a user to scrape any website with different parameters while bypassing most of the anti-bot tools. Check out the [Oxylabs documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api/other-websites) to get the full list of parameters.

Claim free API credentials by creating an Oxylabs account [here](https://oxylabs.io/).
"""
logger.info("# Using OxylabsWebReader")



reader = OxylabsWebReader(
    username="OXYLABS_USERNAME", password="OXYLABS_PASSWORD"
)

documents = reader.load_data(
    [
        "https://sandbox.oxylabs.io/products/1",
        "https://sandbox.oxylabs.io/products/2",
    ]
)

logger.debug(documents[0].text)

"""
Another example with parameters for selecting the geolocation, user agent type, JavaScript rendering, headers, and cookies.
"""
logger.info("Another example with parameters for selecting the geolocation, user agent type, JavaScript rendering, headers, and cookies.")

documents = reader.load_data(
    [
        "https://sandbox.oxylabs.io/products/3",
    ],
    {
        "geo_location": "Berlin, Germany",
        "render": "html",
        "user_agent_type": "mobile",
        "context": [
            {"key": "force_headers", "value": True},
            {"key": "force_cookies", "value": True},
            {
                "key": "headers",
                "value": {
                    "Content-Type": "text/html",
                    "Custom-Header-Name": "custom header content",
                },
            },
            {
                "key": "cookies",
                "value": [
                    {"key": "NID", "value": "1234567890"},
                    {"key": "1P JAR", "value": "0987654321"},
                ],
            },
            {"key": "http_method", "value": "get"},
            {"key": "follow_redirects", "value": True},
            {"key": "successful_status_codes", "value": [808, 909]},
        ],
    },
)

"""
# Using ZenRows Web Reader 🌐

[ZenRows](https://www.zenrows.com/) is a powerful web scraping API that provides advanced features for bypassing anti-bot measures and extracting data from modern websites.

Key Features:
- **JavaScript Rendering**: Handle SPAs and dynamic content with headless browser rendering
- **Premium Proxies**: Bypass anti-bot protection with 55M+ residential IPs from 190+ countries  
- **Session Management**: Maintain the same IP across multiple requests
- **Advanced Data Extraction**: Use CSS selectors or automatic parsing to extract specific data
- **Multiple Output Formats**: Get results in HTML, Markdown, Text, or PDF format
- **Geolocation Support**: Use proxies from specific countries for geo-restricted content

**Prerequisites:** You need to have a ZenRows API key to use this reader. You can get one at [zenrows.com](https://app.zenrows.com/register).
"""
logger.info("# Using ZenRows Web Reader 🌐")


zenrows_reader = ZenRowsWebReader(
    # Get one at https://app.zenrows.com/register
    response_type="markdown",
)

documents = zenrows_reader.load_data(["https://httpbin.io/html"])
logger.debug(documents[0].text[:500])  # Print first 500 characters

zenrows_advanced = ZenRowsWebReader(
    js_render=True,  # Enable JavaScript rendering
    premium_proxy=True,  # Use residential proxies
    proxy_country="us",  # Optional: specify country
)

documents = zenrows_advanced.load_data(
    ["https://www.scrapingcourse.com/antibot-challenge"]
)
logger.debug(f"Scraped {len(documents[0].text)} characters with advanced features")

zenrows_reader = ZenRowsWebReader(
    js_render=True, response_type="markdown"
)

urls = ["https://example.com/", "https://httpbin.io/html"]

documents = zenrows_reader.load_data(urls)

index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What content was found on these pages?")

display(Markdown(f"<b>{response}</b>"))

"""
For more advanced features like custom headers, CSS data extraction, screenshot capabilities, and detailed configuration options, visit the [ZenRows documentation](https://docs.zenrows.com/universal-scraper-api/api-reference).

# Using Olostep Web Reader 🧢

[Olostep](https://www.olostep.com/) is reliable and **cost-effective web scraping API built for scale.** It bypasses bot detection, delivers results in seconds, and can process millions of requests. 

The API returns clean data from any website in various formats, including Markdown, HTML, and structured JSON. 

Sign up [here](https://www.olostep.com/auth) and get 1000 credits for free.
"""
logger.info("# Using Olostep Web Reader 🧢")


reader = OlostepWebReader(mode="scrape")

documents = reader.load_data(url="https://www.olostep.com/")

index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Summarize in 100 words")

logger.debug(response)


reader = OlostepWebReader(mode="search")

documents = reader.load_data(query="What are the latest advancements in AI?")

documents_with_params = reader.load_data(
    query="What are the latest advancements in AI?", params={"country": "US"}
)

index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("List me the headlines")

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)