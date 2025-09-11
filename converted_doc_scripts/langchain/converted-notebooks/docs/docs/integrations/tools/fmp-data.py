from enum import Enum
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentExecutor, create_ollama_functions_agent
from langchain.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from langchain_fmp_data import FMPDataTool
from langchain_fmp_data import FMPDataToolkit
from langchain_fmp_data.tools import ResponseFormat
from typing import Any
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
# FMP Data

Access financial market data through natural language queries.

## Overview

The FMP (Financial Modeling Prep) LangChain integration provides a seamless way to access financial market data through natural language queries. This integration offers two main components:

- `FMPDataToolkit`: Creates collections of tools based on natural language queries
- `FMPDataTool`: A single unified tool that automatically selects and uses the appropriate endpoints

The integration leverages LangChain's semantic search capabilities to match user queries with the most relevant FMP API endpoints, making financial data access more intuitive and efficient.
## Setup
"""
logger.info("# FMP Data")

# !pip install -U langchain-fmp-data


os.environ["FMP_API_KEY"] = "your-fmp-api-key"  # pragma: allowlist secret
# os.environ["OPENAI_API_KEY"] = "your-ollama-api-key"  # pragma: allowlist secret

"""
It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:
"""
logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:")



"""
## Instantiation
There are two main ways to instantiate the FMP LangChain integration:
1. Using FMPDataToolkit
"""
logger.info("## Instantiation")


query = "Get stock market prices and technical indicators"
toolkit = FMPDataToolkit(query=query)

market_toolkit = FMPDataToolkit(
    query=query,
    num_results=5,
)

custom_toolkit = FMPDataToolkit(
    query="Financial analysis",
    num_results=3,
    similarity_threshold=0.4,
    cache_dir="/custom/cache/path",
)

"""
2. Using FMPDataTool
"""
logger.info("2. Using FMPDataTool")


tool = FMPDataTool()

advanced_tool = FMPDataTool(
    max_iterations=50,
    temperature=0.2,
)

"""
## Invocation
The tools can be invoked in several ways:

### Direct Invocation
"""
logger.info("## Invocation")

tool_direct = FMPDataTool()

result = tool.invoke({"query": "What's Apple's current stock price?"})

detailed_result = tool_direct.invoke(
    {
        "query": "Compare Tesla and Ford's profit margins",
        "response_format": ResponseFormat.BOTH,
    }
)

"""
### Using with LangChain Agents
"""
logger.info("### Using with LangChain Agents")


llm = ChatOllama(model="llama3.2")
toolkit = FMPDataToolkit(
    query="Stock analysis",
    num_results=3,
)
tools = toolkit.get_tools()

prompt = "You are a helpful assistant. Answer the user's questions based on the provided context."
agent = create_ollama_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
)

response = agent_executor.invoke({"input": "What's the PE ratio of Microsoft?"})

"""
## Advanced Usage
You can customize the tool's behavior:
"""
logger.info("## Advanced Usage")

advanced_tool = FMPDataTool(
    max_iterations=50,  # Increase max iterations for complex queries
    temperature=0.2,  # Adjust temperature for more/less focused responses
)

query = """
Analyze Apple's financial health by:
1. Examining current ratios and debt levels
2. Comparing profit margins to industry average
3. Looking at cash flow trends
4. Assessing growth metrics
"""
response = advanced_tool.invoke(
    {
        "query": query,
        "response_format": ResponseFormat.BOTH}
)
logger.debug("Detailed Financial Analysis:")
logger.debug(response)

"""
## Chaining
You can chain the tool similar to other tools simply by creating a chain with desired model.
"""
logger.info("## Chaining")


llm = ChatOllama(model="llama3.2")
toolkit = FMPDataToolkit(query="Stock analysis", num_results=3)
tools = toolkit.get_tools()

llm_with_tools = llm.bind(functions=tools)
output_parser = StrOutputParser()
runner = llm_with_tools | output_parser

response = runner.invoke(
    {
        "input": "What's the PE ratio of Microsoft?"
    }
)

"""
## API reference

### FMPDataToolkit
Main class for creating collections of FMP API tools:
"""
logger.info("## API reference")




class FMPDataToolkit:
    """Creates a collection of FMP data tools based on queries."""

    def __init__(
        self,
        query: str | None = None,
        num_results: int = 3,
        similarity_threshold: float = 0.3,
        cache_dir: str | None = None,
    ): ...

    def get_tools(self) -> list[Tool]:
        """Returns a list of relevant FMP API tools based on the query."""
        ...

"""
### FMPDataTool
Unified tool that automatically selects appropriate FMP endpoints:
"""
logger.info("### FMPDataTool")

class FMPDataTool:
    """Single unified tool for accessing FMP data through natural language."""

    def __init__(
            self,
            max_iterations: int = 3,
            temperature: float = 0.0,
    ): ...

    def invoke(
            self,
            input: dict[str, Any],
    ) -> str | dict[str, Any]:
        """Execute a natural language query against FMP API."""
        ...

"""
### ResponseFormat
Enum for controlling response format:
"""
logger.info("### ResponseFormat")



class ResponseFormat(str, Enum):
    RAW = "raw"  # Raw API response
    ANALYSIS = "text"  # Natural language analysis
    BOTH = "both"  # Both raw data and analysis

logger.info("\n\n[DONE]", bright=True)