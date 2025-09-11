from jet.logger import logger
from langchain_agentql.document_loaders import AgentQLLoader
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
# AgentQLLoader

[AgentQL](https://www.agentql.com/)'s document loader provides structured data extraction from any web page using an [AgentQL query](https://docs.agentql.com/agentql-query). AgentQL can be used across multiple languages and web pages without breaking over time and change.

## Overview

`AgentQLLoader` requires the following two parameters:
- `url`: The URL of the web page you want to extract data from.
- `query`: The AgentQL query to execute. Learn more about [how to write an AgentQL query in the docs](https://docs.agentql.com/agentql-query) or test one out in the [AgentQL Playground](https://dev.agentql.com/playground).

Setting the following parameters are optional:
- `api_key`: Your AgentQL API key from [dev.agentql.com](https://dev.agentql.com). **`Optional`.**
- `timeout`: The number of seconds to wait for a request before timing out. **Defaults to `900`.**
- `is_stealth_mode_enabled`: Whether to enable experimental anti-bot evasion strategies. This feature may not work for all websites at all times. Data extraction may take longer to complete with this mode enabled. **Defaults to `False`.**
- `wait_for`: The number of seconds to wait for the page to load before extracting data. **Defaults to `0`.**
- `is_scroll_to_bottom_enabled`: Whether to scroll to bottom of the page before extracting data. **Defaults to `False`.**
- `mode`: `"standard"` uses deep data analysis, while `"fast"` trades some depth of analysis for speed and is adequate for most usecases. [Learn more about the modes in this guide.](https://docs.agentql.com/accuracy/standard-mode) **Defaults to `"fast"`.**
- `is_screenshot_enabled`: Whether to take a screenshot before extracting data. Returned in 'metadata' as a Base64 string. **Defaults to `False`.**

AgentQLLoader is implemented with AgentQL's [REST API](https://docs.agentql.com/rest-api/api-reference)

### Integration details

| Class | Package | Local | Serializable | JS support |
| :--- | :--- | :---: | :---: |  :---: |
| AgentQLLoader| langchain-agentql | ✅ | ❌ | ❌ |

### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: |
| AgentQLLoader | ✅ | ❌ |

## Setup

To use the AgentQL Document Loader, you will need to configure the `AGENTQL_API_KEY` environment variable, or use the `api_key` parameter. You can acquire an API key from our [Dev Portal](https://dev.agentql.com).

### Installation

Install **langchain-agentql**.
"""
logger.info("# AgentQLLoader")

# %pip install -qU langchain-agentql

"""
### Set Credentials
"""
logger.info("### Set Credentials")


os.environ["AGENTQL_API_KEY"] = "YOUR_AGENTQL_API_KEY"

"""
## Initialization

Next instantiate your model object:
"""
logger.info("## Initialization")


loader = AgentQLLoader(
    url="https://www.agentql.com/blog",
    query="""
    {
        posts[] {
            title
            url
            date
            author
        }
    }
    """,
    is_scroll_to_bottom_enabled=True,
)

"""
## Load
"""
logger.info("## Load")

docs = loader.load()
docs[0]

logger.debug(docs[0].metadata)

"""
## Lazy Load

`AgentQLLoader` currently only loads one `Document` at a time. Therefore, `load()` and `lazy_load()` behave the same:
"""
logger.info("## Lazy Load")

pages = [doc for doc in loader.lazy_load()]
pages

"""
## API reference

For more information on how to use this integration, please refer to the [git repo](https://github.com/tinyfish-io/agentql-integrations/tree/main/langchain) or the [langchain integration documentation](https://docs.agentql.com/integrations/langchain)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)