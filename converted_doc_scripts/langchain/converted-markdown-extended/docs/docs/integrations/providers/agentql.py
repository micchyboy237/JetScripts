from jet.logger import logger
from langchain_agentql import AgentQLBrowserToolkit
from langchain_agentql.document_loaders import AgentQLLoader
from langchain_agentql.tools import ExtractWebDataTool, ExtractWebDataBrowserTool, GetWebElementBrowserTool
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
# AgentQL

[AgentQL](https://www.agentql.com/) provides web interaction and structured data extraction from any web page using an [AgentQL query](https://docs.agentql.com/agentql-query) or a Natural Language prompt. AgentQL can be used across multiple languages and web pages without breaking over time and change.

## Installation and Setup

Install the integration package:
"""
logger.info("# AgentQL")

pip install langchain-agentql

"""
## API Key

Get an API Key from our [Dev Portal](https://dev.agentql.com/) and add it to your environment variables:

export AGENTQL_API_KEY="your-api-key-here"

## DocumentLoader
AgentQL's document loader provides structured data extraction from any web page using an AgentQL query.
"""
logger.info("## API Key")


"""
See our [document loader documentation and usage example](/docs/integrations/document_loaders/agentql).

## Tools and Toolkits
AgentQL tools provides web interaction and structured data extraction from any web page using an AgentQL query or a Natural Language prompt.
"""
logger.info("## Tools and Toolkits")


"""
See our [tools documentation and usage example](/docs/integrations/tools/agentql).
"""
logger.info("See our [tools documentation and usage example](/docs/integrations/tools/agentql).")

logger.info("\n\n[DONE]", bright=True)