import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.agentql import AgentQLBrowserToolSpec
from llama_index.tools.agentql import AgentQLRestAPIToolSpec
from llama_index.tools.playwright.base import PlaywrightToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Building a Browser Agent with AgentQL

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-agentql/examples/agentql_browser_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

[AgentQL](https://www.agentql.com/) tools provide web interaction and structured data extraction from any web page using an [AgentQL query](https://docs.agentql.com/agentql-query) or a Natural Language prompt. AgentQL can be used across multiple languages and web pages without breaking over time and change.

This tutorial shows you how to:

* Create a browser agent with AgentQL tools and LlamaIndex
* How to use AgentQL tools to navigate the Internet 
* How to use AgentQL tools to scrape content from the Internet

## Overview

AgentQL provides three function tools. The first doesn't require a browser and relies on the REST API:

- **`extract_web_data_with_rest_api`** extracts structured data as JSON from a web page given a URL using either an [AgentQL query](https://docs.agentql.com/agentql-query/query-intro) or a Natural Language description of the data.

The other two tools must be used with a `Playwright` browser or a remote browser instance via Chrome DevTools Protocal (CDP):

- **`extract_web_data_from_browser`** extracts structured data as JSON from the active web page in a browser using either an [AgentQL query](https://docs.agentql.com/agentql-query/query-intro) or a Natural Language description.

- **`get_web_element_from_browser`** finds a web element on the active web page in a browser using a Natural Language description and returns its CSS selector for further interaction.

### Tool features

| Tool | Web Data Extraction | Web Element Extraction | Use With Local Browser |
| :--- | :---: | :---: | :---: |
| extract_web_data_with_rest_api | ✅ | ❌ | ❌
| extract_web_data_from_browser | ✅ | ❌ | ✅
| get_web_element_from_browser | ❌ | ✅ | ✅

## Set up
"""
logger.info("# Building a Browser Agent with AgentQL")

# %pip install llama-index-tools-agentql llama-index-tools-playwright llama-index

"""
### Credentials

To use the AgentQL tools, you will need to get your own API key from the [AgentQL Dev Portal](https://dev.agentql.com/) and set the AgentQL environment variable:
"""
logger.info("### Credentials")


os.environ["AGENTQL_API_KEY"] = "YOUR_AGENTQL_API_KEY"

"""
### Set up Playwright browser and AgentQL tools
To run this notebook, install Playwright browser and configure Jupyter Notebook's `asyncio` loop.
"""
logger.info("### Set up Playwright browser and AgentQL tools")

# !playwright install

# import nest_asyncio

# nest_asyncio.apply()

"""
## Instantiation

### `AgentQLRestAPIToolSpec`
`AgentQLRestAPIToolSpec` provides `extract_web_data_with_rest_api` function tool.

You can instantiate `AgentQLRestAPIToolSpec` with the following param:
- `timeout`: The number of seconds to wait for a request before timing out. Increase if data extraction times out. **Defaults to `900`.**
- `is_stealth_mode_enabled`: Whether to enable experimental anti-bot evasion strategies. This feature may not work for all websites at all times. Data extraction may take longer to complete with this mode enabled. **Defaults to `False`.**
- `wait_for`: The number of seconds to wait for the page to load before extracting data. **Defaults to `0`.**
- `is_scroll_to_bottom_enabled`: Whether to scroll to bottom of the page before extracting data. **Defaults to `False`.**
- `mode`: `"standard"` uses deep data analysis, while `"fast"` trades some depth of analysis for speed and is adequate for most usecases. [Learn more about the modes in this guide.](https://docs.agentql.com/accuracy/standard-mode) **Defaults to `"fast"`.**
- `is_screenshot_enabled`: Whether to take a screenshot before extracting data. Returned in 'metadata' as a Base64 string. **Defaults to `False`.**

`AgentQLRestAPIToolSpec` is using AgentQL REST API, for more details about the parameters read [API Reference docs](https://docs.agentql.com/rest-api/api-reference).
"""
logger.info("## Instantiation")


agentql_rest_api_tool = AgentQLRestAPIToolSpec()

"""
### `AgentQLBrowserToolSpec`
`AgentQLBrowserToolSpec` provides 2 tools: `extract_web_data_from_browser` and `get_web_element_from_browser`.

This tool spec can be instantiated with the following params:
- `async_browser`: An async playwright browser instance.
- `timeout_for_data`: The number of seconds to wait for a extract data request before timing out. **Defaults to `900`.**
- `timeout_for_element`: The number of seconds to wait for a get element request before timing out. **Defaults to `900`.**
- `wait_for_network_idle`: Whether to wait until the network reaches a full idle state before executing. **Defaults to `True`.**
- `include_hidden_for_data`: Whether to take into account visually hidden elements on the page for extract data. **Defaults to `True`.**
- `include_hidden_for_element`: Whether to take into account visually hidden elements on the page for get element. **Defaults to `False`.**
- `mode`: `"standard"` uses deep data analysis, while `"fast"` trades some depth of analysis for speed and is adequate for most usecases. [Learn more about the modes in this guide.](https://docs.agentql.com/accuracy/standard-mode) **Defaults to `"fast"`.**

`AgentQLBrowserToolSpec` is using AgentQL SDK. You can find more details about the parameters and the functions in [SDK API Reference](https://docs.agentql.com/python-sdk/api-references/agentql-page).

> **Note:** To instantiate `AgentQLBrowserToolSpec` you need to provide a browser instance. You can create one using  `create_async_playwright_browser` utility method from LlamaIndex's Playwright ToolSpec.
"""
logger.info("### `AgentQLBrowserToolSpec`")


async def run_async_code_d44d8bdd():
    async def run_async_code_814c35e8():
        async_browser = await PlaywrightToolSpec.create_async_playwright_browser()
        return async_browser
    async_browser = asyncio.run(run_async_code_814c35e8())
    logger.success(format_json(async_browser))
    return async_browser
async_browser = asyncio.run(run_async_code_d44d8bdd())
logger.success(format_json(async_browser))
agentql_browser_tool = AgentQLBrowserToolSpec(async_browser=async_browser)

"""
## Invoking the AgentQL tools

### `extract_web_data_with_rest_api`

This tool uses AgentQL's REST API under the hood, sending the publically available web page's URL to AgentQL's endpoint. This will not work with private pages or logged in sessions. Use `extract_web_data_from_browser` for those usecases.

- `url`: The URL of the web page you want to extract data from.
- `query`: The AgentQL query to execute. Use this if you want to extract data in a structure you define. Learn more about [how to write an AgentQL query in the docs](https://docs.agentql.com/agentql-query).
- `prompt`: A Natural Language description of the data to extract from the page. AgentQL will infer the data’s structure from your prompt.

> **Note:** You must define either a `query` or a `prompt` to use AgentQL.
"""
logger.info("## Invoking the AgentQL tools")

await agentql_rest_api_tool.extract_web_data_with_rest_api(
    url="https://www.agentql.com/blog",
    query="{ posts[] { title url author date }}",
)

"""
#### Stealth Mode
AgentQL provides experimental anti-bot evasion strategies to avoid detection by anti-bot services.

> **Note**: Stealth mode is experimental and may not work for all websites at all times. The data extraction may take longer to complete comparing to non-stealth mode.
"""
logger.info("#### Stealth Mode")

await agentql_rest_api_tool.extract_web_data_with_rest_api(
    url="https://www.patagonia.com/shop/web-specials/womens",
    query="{ items[] { name price}}",
)

"""
### `extract_web_data_from_browser`

- `query`: The AgentQL query to execute. Use this if you want to extract data in a structure you define. Learn more about [how to write an AgentQL query in the docs](https://docs.agentql.com/agentql-query).
- `prompt`: A Natural Language description of the data to extract from the page. AgentQL will infer the data’s structure from your prompt.

> **Note:** You must define either a `query` or a `prompt` to use AgentQL.

To extract data, first you must navigate to a web page using LlamaIndex's [Playwright](https://docs.llamaindex.ai/en/stable/api_reference/tools/playwright/) click tool.
"""
logger.info("### `extract_web_data_from_browser`")

playwright_tool = PlaywrightToolSpec(async_browser=async_browser)
async def run_async_code_bfa4b1b8():
    await playwright_tool.navigate_to("https://www.agentql.com/blog")
    return 
 = asyncio.run(run_async_code_bfa4b1b8())
logger.success(format_json())



await agentql_browser_tool.extract_web_data_from_browser(
    prompt="the blog posts with title and url",
)

"""
### `get_web_element_from_browser`

- `prompt`: A Natural Language description of the web element to find on the page.
"""
logger.info("### `get_web_element_from_browser`")

async def run_async_code_bfa4b1b8():
    await playwright_tool.navigate_to("https://www.agentql.com/blog")
    return 
 = asyncio.run(run_async_code_bfa4b1b8())
logger.success(format_json())
async def run_async_code_30e8a2f7():
    logger.debug(await playwright_tool.get_current_page())
    return 
 = asyncio.run(run_async_code_30e8a2f7())
logger.success(format_json())

async def async_func_3():
    next_page_button = await agentql_browser_tool.get_web_element_from_browser(
        prompt="The next page navigation button",
    )
    return next_page_button
next_page_button = asyncio.run(async_func_3())
logger.success(format_json(next_page_button))
next_page_button

"""
Click on the element and check the url again
"""
logger.info("Click on the element and check the url again")

async def run_async_code_32960de2():
    await playwright_tool.click(next_page_button)
    return 
 = asyncio.run(run_async_code_32960de2())
logger.success(format_json())

async def run_async_code_30e8a2f7():
    logger.debug(await playwright_tool.get_current_page())
    return 
 = asyncio.run(run_async_code_30e8a2f7())
logger.success(format_json())

"""
## Using the AgentQL tools with agent
To get started, you will need an [MLX api key](https://platform.openai.com/account/api-keys)
"""
logger.info("## Using the AgentQL tools with agent")


# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


playwright_tool = PlaywrightToolSpec(async_browser=async_browser)
playwright_tool_list = playwright_tool.to_tool_list()
playwright_agent_tool_list = [
    tool
    for tool in playwright_tool_list
    if tool.metadata.name in ["click", "get_current_page", "navigate_to"]
]

agent = FunctionAgent(
    tools=playwright_agent_tool_list + agentql_browser_tool.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit"),
)

logger.debug(
    await agent.run(
        """
        Navigate to https://blog.samaltman.com/archive,
        Find blog posts titled "What I wish someone had told me", click on the link,
        Extract the blog text and number of views.
        """
    )
)

logger.info("\n\n[DONE]", bright=True)