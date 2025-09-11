from jet.logger import logger
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import SerpAPIWrapper
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
# SerpAPI

This page covers how to use the SerpAPI search APIs within LangChain.
It is broken into two parts: installation and setup, and then references to the specific SerpAPI wrapper.

## Installation and Setup
- Install requirements with `pip install google-search-results`
- Get a SerpAPI api key and either set it as an environment variable (`SERPAPI_API_KEY`)

## Wrappers

### Utility

There exists a SerpAPI utility which wraps this API. To import this utility:
"""
logger.info("# SerpAPI")


"""
For a more detailed walkthrough of this wrapper, see [this notebook](/docs/integrations/tools/serpapi).

### Tool

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:
"""
logger.info("### Tool")

tools = load_tools(["serpapi"])

"""
For more information on this, see [this page](/docs/how_to/tools_builtin)
"""
logger.info("For more information on this, see [this page](/docs/how_to/tools_builtin)")

logger.info("\n\n[DONE]", bright=True)