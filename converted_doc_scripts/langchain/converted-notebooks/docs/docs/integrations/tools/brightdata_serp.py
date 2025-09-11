from jet.logger import logger
from langchain_brightdata import BrightDataSERP
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
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
# BrightDataSERP

[Bright Data](https://brightdata.com/) provides a powerful SERP API that allows you to query search engines (Google,Bing.DuckDuckGo,Yandex) with geo-targeting and advanced customization options, particularly useful for AI agents requiring real-time web information.


## Overview

### Integration details


|Class|Package|Serializable|JS support|Package latest|
|:--|:--|:-:|:-:|:-:|
|[BrightDataSERP](https://pypi.org/project/langchain-brightdata/)|[langchain-brightdata](https://pypi.org/project/langchain-brightdata/)|✅|❌|![PyPI - Version](https://img.shields.io/pypi/v/langchain-brightdata?style=flat-square&label=%20)|


### Tool features


|Native async|Returns artifact|Return data|Pricing|
|:-:|:-:|:--|:-:|
|❌|❌|Title, URL, snippet, position, and other search result data|Requires Bright Data account|



## Setup

The integration lives in the `langchain-brightdata` package.

pip install langchain-brightdata

### Credentials

You'll need a Bright Data API key to use this tool. You can set it as an environment variable:
"""
logger.info("# BrightDataSERP")


os.environ["BRIGHT_DATA_API_KEY"] = "your-api-key"

"""
Or pass it directly when initializing the tool:
"""
logger.info("Or pass it directly when initializing the tool:")


serp_tool = BrightDataSERP(bright_data_)

"""
## Instantiation

Here we show how to instantiate an instance of the BrightDataSERP tool. This tool allows you to perform search engine queries with various customization options including geo-targeting, language preferences, device type simulation, and specific search types using Bright Data's SERP API.

The tool accepts various parameters during instantiation:

- `bright_data_api_key` (required, str): Your Bright Data API key for authentication.
- `search_engine` (optional, str): Search engine to use for queries. Default is "google". Other options include "bing", "yahoo", "yandex", "DuckDuckGo" etc.
- `country` (optional, str): Two-letter country code for localized search results (e.g., "us", "gb", "de", "jp"). Default is "us".
- `language` (optional, str): Two-letter language code for the search results (e.g., "en", "es", "fr", "de"). Default is "en".
- `results_count` (optional, int): Number of search results to return. Default is 10. Maximum value is typically 100.
- `search_type` (optional, str): Type of search to perform. Options include:
    - None (default): Regular web search
    - "isch": Images search
    - "shop": Shopping search
    - "nws": News search
    - "jobs": Jobs search
- `device_type` (optional, str): Device type to simulate for the search. Options include:
    - None (default): Desktop device
    - "mobile": Generic mobile device
    - "ios": iOS device (iPhone)
    - "android": Android device
- `parse_results` (optional, bool): Whether to return parsed JSON results. Default is False, which returns raw HTML response.

## Invocation

### Basic Usage
"""
logger.info("## Instantiation")


serp_tool = BrightDataSERP(
    bright_data_  # Optional if set in environment variables
)

results = serp_tool.invoke("latest AI research papers")

logger.debug(results)

"""
### Advanced Usage with Parameters
"""
logger.info("### Advanced Usage with Parameters")


serp_tool = BrightDataSERP(
    bright_data_search_engine="google",  # Default
    country="us",  # Default
    language="en",  # Default
    results_count=10,  # Default
    parse_results=True,  # Get structured JSON results
)

results = serp_tool.invoke(
    {
        "query": "best electric vehicles",
        "country": "de",  # Get results as if searching from Germany
        "language": "de",  # Get results in German
        "search_type": "shop",  # Get shopping results
        "device_type": "mobile",  # Simulate a mobile device
        "results_count": 15,
    }
)

logger.debug(results)

"""
## Customization Options

The BrightDataSERP tool accepts several parameters for customization:

|Parameter|Type|Description|
|:--|:--|:--|
|`query`|str|The search query to perform|
|`search_engine`|str|Search engine to use (default: "google")|
|`country`|str|Two-letter country code for localized results (default: "us")|
|`language`|str|Two-letter language code (default: "en")|
|`results_count`|int|Number of results to return (default: 10)|
|`search_type`|str|Type of search: None (web), "isch" (images), "shop", "nws" (news), "jobs"|
|`device_type`|str|Device type: None (desktop), "mobile", "ios", "android"|
|`parse_results`|bool|Whether to return structured JSON (default: False)|

## Use within an agent
"""
logger.info("## Customization Options")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_)

serp_tool = BrightDataSERP(
    bright_data_search_engine="google",
    country="us",
    language="en",
    results_count=10,
    parse_results=True,
)

agent = create_react_agent(llm, [serp_tool])

user_input = "Search for 'best electric vehicles' shopping results in Germany in German using mobile."

for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## API reference

- [Bright Data API Documentation](https://docs.brightdata.com/scraping-automation/serp-api/introduction)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)