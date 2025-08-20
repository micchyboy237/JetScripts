import asyncio
from jet.transformers.formatters import format_json
from google.colab import userdata
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import (
AgentStream,
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.tools.agentql import AgentQLBrowserToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.tools.playwright.base import PlaywrightToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Agent Workflow + Research Assistant using AgentQL

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/agent_workflow_research_assistant.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In this tutorial, we will use an `AgentWorkflow` to build a research assistant MLX agent using tools including AgentQL's browser tools, Playwright's tools, and the DuckDuckGoSearch tool. This agent performs a web search to find relevant resources for a research topic, interacts with them, and extracts key metadata (e.g., title, author, publication details, and abstract) from those resources.

## Initial Setup

The main things we need to get started are:

- <a href="https://platform.openai.com/api-keys" target="_blank">MLX's API key</a>
- <a href="https://dev.agentql.com/api-keys" target="_blank">AgentQL's API key</a>

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙 and Playwright.
"""
logger.info("# Agent Workflow + Research Assistant using AgentQL")

# %pip install llama-index
# %pip install llama-index-tools-agentql
# %pip install llama-index-tools-playwright
# %pip install llama-index-tools-duckduckgo

# !playwright install

"""
# Store your `OPENAI_API_KEY` and `AGENTQL_API_KEY` keys in <a href="https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75" target="_blank">Google Colab's secrets</a>.
"""
# logger.info("Store your `OPENAI_API_KEY` and `AGENTQL_API_KEY` keys in <a href="https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75" target="_blank">Google Colab's secrets</a>.")



os.environ["AGENTQL_API_KEY"] = userdata.get("AGENTQL_API_KEY")
# os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

"""
Let's start by enabling async for the notebook since an online environment like Google Colab only supports an asynchronous version of AgentQL.
"""
logger.info("Let's start by enabling async for the notebook since an online environment like Google Colab only supports an asynchronous version of AgentQL.")

# import nest_asyncio

# nest_asyncio.apply()

"""
Create an `async_browser` instance and select the <a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/playwright/" target="_blank">Playwright tools</a> you want to use.
"""
logger.info("Create an `async_browser` instance and select the <a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/playwright/" target="_blank">Playwright tools</a> you want to use.")


async def async_func_2():
    async_browser = await PlaywrightToolSpec.create_async_playwright_browser(
        headless=True
    )
    return async_browser
async_browser = asyncio.run(async_func_2())
logger.success(format_json(async_browser))

playwright_tool = PlaywrightToolSpec(async_browser=async_browser)
playwright_tool_list = playwright_tool.to_tool_list()
playwright_agent_tool_list = [
    tool
    for tool in playwright_tool_list
    if tool.metadata.name in ["click", "get_current_page", "navigate_to"]
]

"""
Import the <a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/agentql/" target="_blank">AgentQL browser tools</a> and <a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/duckduckgo/" target="_blank">DuckDuckGo full search tool</a>.
"""
logger.info("Import the <a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/agentql/" target="_blank">AgentQL browser tools</a> and <a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/duckduckgo/" target="_blank">DuckDuckGo full search tool</a>.")


duckduckgo_search_tool = [
    tool
    for tool in DuckDuckGoSearchToolSpec().to_tool_list()
    if tool.metadata.name == "duckduckgo_full_search"
]

agentql_browser_tool = AgentQLBrowserToolSpec(async_browser=async_browser)

"""
We can now create an `AgentWorkFlow` that uses the tools that we have imported.
"""
logger.info("We can now create an `AgentWorkFlow` that uses the tools that we have imported.")


llm = MLX(model="qwen3-1.7b-4bit")

workflow = AgentWorkflow.from_tools_or_functions(
    playwright_agent_tool_list
    + agentql_browser_tool.to_tool_list()
    + duckduckgo_search_tool,
    llm=llm,
    system_prompt="You are an expert that can do browser automation, data extraction and text summarization for finding and extracting data from research resources.",
)

"""
`AgentWorkflow` also supports streaming, which works by using the handler that is returned from the workflow. To stream the LLM output, you can use the `AgentStream` events.
"""


handler = workflow.run(
    user_msg="""
    Use DuckDuckGoSearch to find URL resources on the web that are relevant to the research topic: What is the relationship between exercise and stress levels?
    Go through each resource found. For each different resource, use Playwright to click on link to the resource, then use AgentQL to extract information, including the name of the resource, author name(s), link to the resource, publishing date, journal name, volume number, issue number, and the abstract.
    Find more resources until there are two different resources that can be successfully extracted from.
    """
)

async for event in handler.stream_events():
    if isinstance(event, AgentStream):
        logger.debug(event.delta, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)