from jet.logger import logger
from langchain_oxylabs import OxylabsLoader
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
# Oxylabs

[Oxylabs](https://oxylabs.io/) is a web intelligence collection platform that enables companies worldwide to unlock data-driven insights.

## Overview

Oxylabs document loader allows to load data from search engines, e-commerce sites, travel platforms, and any other website. It supports geolocation, browser rendering, data parsing, multiple user agents and many more parameters. Check out [Oxylabs documentation](https://developers.oxylabs.io/scraping-solutions/web-scraper-api) for more information.


### Integration details

| Class         | Package                                                           | Local | Serializable |            Pricing            |
|:--------------|:------------------------------------------------------------------|:-----:|:------------:|:-----------------------------:|
| OxylabsLoader | [langchain-oxylabs](https://github.com/oxylabs/langchain-oxylabs) |   ✅   |      ❌       | Free 5,000 results for 1 week |

### Loader features
| Document Lazy Loading |
|:---------------------:|
|           ✅           |

## Setup

Install the required dependencies.
"""
logger.info("# Oxylabs")

# %pip install -U langchain-oxylabs

"""
### Credentials

Set up the proper API keys and environment variables.
Create your API user credentials: Sign up for a free trial or purchase the product
in the [Oxylabs dashboard](https://dashboard.oxylabs.io/en/registration)
to create your API user credentials (OXYLABS_USERNAME and OXYLABS_PASSWORD).
"""
logger.info("### Credentials")

# import getpass

# os.environ["OXYLABS_USERNAME"] = getpass.getpass("Enter your Oxylabs username: ")
# os.environ["OXYLABS_PASSWORD"] = getpass.getpass("Enter your Oxylabs password: ")

"""
## Initialization
"""
logger.info("## Initialization")


loader = OxylabsLoader(
    urls=[
        "https://sandbox.oxylabs.io/products/1",
        "https://sandbox.oxylabs.io/products/2",
    ],
    params={"markdown": True},
)

"""
#
#
 
L
o
a
d
"""
logger.info("#")

for document in loader.load():
    logger.debug(document.page_content[:1000])

"""
#
#
 
L
a
z
y
 
L
o
a
d
"""
logger.info("#")

for document in loader.lazy_load():
    logger.debug(document.page_content[:1000])

"""
## Advanced examples

The following examples show the usage of `OxylabsLoader` with geolocation, currency, pagination and user agent parameters for Amazon Search and Google Search sources.
"""
logger.info("## Advanced examples")

loader = OxylabsLoader(
    queries=["gaming headset", "gaming chair", "computer mouse"],
    params={
        "source": "amazon_search",
        "parse": True,
        "geo_location": "DE",
        "currency": "EUR",
        "pages": 3,
    },
)

loader = OxylabsLoader(
    queries=["europe gdp per capita", "us gdp per capita"],
    params={
        "source": "google_search",
        "parse": True,
        "geo_location": "Paris, France",
        "user_agent_type": "mobile",
    },
)

"""
## API reference

[More information about this package.](https://github.com/oxylabs/langchain-oxylabs)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)