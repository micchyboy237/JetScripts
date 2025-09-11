from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_ollama_functions_agent
from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
from langchain_community.tools.polygon.aggregates import PolygonAggregates
from langchain_community.tools.polygon.financials import PolygonFinancials
from langchain_community.tools.polygon.last_quote import PolygonLastQuote
from langchain_community.tools.polygon.ticker_news import PolygonTickerNews
from langchain_community.utilities.polygon import PolygonAPIWrapper
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
# Polygon IO Toolkit and Tools

This notebook shows how to use agents to interact with the [Polygon IO](https://polygon.io/) toolkit. The toolkit provides access to Polygon's Stock Market Data API.

## Setup

### Installation

To use Polygon IO tools, you need to install the `langchain-community` package.
"""
logger.info("# Polygon IO Toolkit and Tools")

# %pip install -qU langchain-community > /dev/null

"""
### Credentials

Get your Polygon IO API key [here](https://polygon.io/), and then set it below.
"""
logger.info("### Credentials")

# import getpass

if "POLYGON_API_KEY" not in os.environ:
#     os.environ["POLYGON_API_KEY"] = getpass.getpass()

"""
It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability
"""
logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability")



"""
## Toolkit

Polygon IO provides both a toolkit and individual tools for each of the tools included in the toolkit. Let's first explore using the toolkit and then we will walk through using the individual tools.

### Initialization

We can initialize the toolkit by importing it alongside the API wrapper needed to use the tools.
"""
logger.info("## Toolkit")


polygon = PolygonAPIWrapper()
toolkit = PolygonToolkit.from_polygon_api_wrapper(polygon)

"""
### Tools

We can examine the tools included in this toolkit:
"""
logger.info("### Tools")

toolkit.get_tools()

"""
### Use within an agent

Next we can add our toolkit to an agent and use it!
"""
logger.info("### Use within an agent")


llm = ChatOllama(model="llama3.2")

instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/ollama-functions-template")
prompt = base_prompt.partial(instructions=instructions)

agent = create_ollama_functions_agent(llm, toolkit.get_tools(), prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

"""
We can examine yesterdays information for a certain ticker:
"""
logger.info("We can examine yesterdays information for a certain ticker:")

agent_executor.invoke({"input": "What was yesterdays financial info for AAPL?"})

"""
We can also ask for recent news regarding a stock:
"""
logger.info("We can also ask for recent news regarding a stock:")

agent_executor.invoke({"input": "What is the recent new regarding MSFT?"})

"""
You can also ask about financial information for a company:
"""
logger.info("You can also ask about financial information for a company:")

agent_executor.invoke(
    {"input": "What were last quarters financial numbers for Nvidia?"}
)

"""
Lastly, you can get live data, although this requires a "Stocks Advanced" subscription
"""
logger.info("Lastly, you can get live data, although this requires a "Stocks Advanced" subscription")

agent_executor.invoke({"input": "What is Doordash stock price right now?"})

"""
### API reference

For detailed documentation of all the Polygon IO toolkit features and configurations head to the API reference: https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.polygon.toolkit.PolygonToolkit.html

## Tools

First, let's set up the API wrapper that we will use for all the tools and then we will walk through each one of them.
"""
logger.info("### API reference")


api_wrapper = PolygonAPIWrapper()

"""
### Aggregate

This tool shows aggregate information for a stock.
"""
logger.info("### Aggregate")


aggregate_tool = PolygonAggregates(api_wrapper=api_wrapper)

res = aggregate_tool.invoke(
    {
        "ticker": "AAPL",
        "timespan": "day",
        "timespan_multiplier": 1,
        "from_date": "2024-08-01",
        "to_date": "2024-08-07",
    }
)

model_generated_tool_call = {
    "args": {
        "ticker": "AAPL",
        "timespan": "day",
        "timespan_multiplier": 1,
        "from_date": "2024-08-01",
        "to_date": "2024-08-07",
    },
    "id": "1",
    "name": aggregate_tool.name,
    "type": "tool_call",
}

res = aggregate_tool.invoke(model_generated_tool_call)

logger.debug(res)

"""
### Financials

This tool provides general financial information about a stock
"""
logger.info("### Financials")


financials_tool = PolygonFinancials(api_wrapper=api_wrapper)

res = financials_tool.invoke({"query": "AAPL"})

model_generated_tool_call = {
    "args": {"query": "AAPL"},
    "id": "1",
    "name": financials_tool.name,
    "type": "tool_call",
}

res = financials_tool.invoke(model_generated_tool_call)

logger.debug(res)

"""
### Last Quote

This tool provides information about the live data of a stock, although it requires a "Stocks Advanced" subscription to use.
"""
logger.info("### Last Quote")


last_quote_tool = PolygonLastQuote(api_wrapper=api_wrapper)

res = last_quote_tool.invoke({"query": "AAPL"})

model_generated_tool_call = {
    "args": {"query": "AAPL"},
    "id": "1",
    "name": last_quote_tool.name,
    "type": "tool_call",
}

res = last_quote_tool.invoke(model_generated_tool_call)

"""
### Ticker News

This tool provides recent news about a certain ticker.
"""
logger.info("### Ticker News")


news_tool = PolygonTickerNews(api_wrapper=api_wrapper)

res = news_tool.invoke({"query": "AAPL"})

model_generated_tool_call = {
    "args": {"query": "AAPL"},
    "id": "1",
    "name": news_tool.name,
    "type": "tool_call",
}

res = news_tool.invoke(model_generated_tool_call)

logger.debug(res)

"""
### API reference

For detailed documentation of all Polygon IO tools head to the API reference for each:

- Aggregate: https://python.langchain.com/api_reference/community/tools/langchain_community.tools.polygon.aggregates.PolygonAggregates.html
- Financials: https://python.langchain.com/api_reference/community/tools/langchain_community.tools.polygon.financials.PolygonFinancials.html
- Last Quote: https://python.langchain.com/api_reference/community/tools/langchain_community.tools.polygon.last_quote.PolygonLastQuote.html
- Ticker News: https://python.langchain.com/api_reference/community/tools/langchain_community.tools.polygon.ticker_news.PolygonTickerNews.html
"""
logger.info("### API reference")

logger.info("\n\n[DONE]", bright=True)