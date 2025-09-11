from jet.logger import logger
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DataForSeoAPISearchResults
from langchain_community.tools import DataForSeoAPISearchRun
from langchain_community.utilities.dataforseo_api_search import DataForSeoAPIWrapper
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
# DataForSEO

>[DataForSeo](https://dataforseo.com/) provides comprehensive SEO and digital marketing data solutions via API.

This page provides instructions on how to use the DataForSEO search APIs within LangChain.

## Installation and Setup

Get a [DataForSEO API Access login and password](https://app.dataforseo.com/register), and set them as environment variables
(`DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` respectively).
"""
logger.info("# DataForSEO")


os.environ["DATAFORSEO_LOGIN"] = "your_login"
os.environ["DATAFORSEO_PASSWORD"] = "your_password"

"""
## Utility

The `DataForSEO` utility wraps the API. To import this utility, use:
"""
logger.info("## Utility")


"""
For a detailed walkthrough of this wrapper, see [this notebook](/docs/integrations/tools/dataforseo).

## Tool

You can also load this wrapper as a Tool to use with an Agent:
"""
logger.info("## Tool")

tools = load_tools(["dataforseo-api-search"])

"""
This will load the following tools:
"""
logger.info("This will load the following tools:")


"""
## Example usage
"""
logger.info("## Example usage")

dataforseo = DataForSeoAPIWrapper(api_login="your_login", api_password="your_password")
result = dataforseo.run("Bill Gates")
logger.debug(result)

logger.info("\n\n[DONE]", bright=True)