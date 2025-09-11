from jet.logger import logger
from langchain_apify import ApifyActorsTool
from langchain_apify import ApifyDatasetLoader
from langchain_apify import ApifyWrapper
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
# Apify


>[Apify](https://apify.com) is a cloud platform for web scraping and data extraction,
>which provides an [ecosystem](https://apify.com/store) of more than a thousand
>ready-made apps called *Actors* for various scraping, crawling, and extraction use cases.

[![Apify Actors](/img/ApifyActors.png)](https://apify.com/store)

This integration enables you run Actors on the `Apify` platform and load their results into LangChain to feed your vector
indexes with documents and data from the web, e.g. to generate answers from websites with documentation,
blogs, or knowledge bases.


## Installation and Setup

- Install the LangChain Apify package for Python with:
"""
logger.info("# Apify")

pip install langchain-apify

"""
- Get your [Apify API token](https://console.apify.com/account/integrations) and either set it as
  an environment variable (`APIFY_API_TOKEN`) or pass it as `apify_api_token` in the constructor.

## Tool

You can use the `ApifyActorsTool` to use Apify Actors with agents.
"""
logger.info("## Tool")


"""
See [this notebook](/docs/integrations/tools/apify_actors) for example usage and a full example of a tool-calling agent with LangGraph in the [Apify LangGraph agent Actor template](https://apify.com/templates/python-langgraph).

For more information on how to use this tool, visit [the Apify integration documentation](https://docs.apify.com/platform/integrations/langgraph).

## Wrapper

You can use the `ApifyWrapper` to run Actors on the Apify platform.
"""
logger.info("## Wrapper")


"""
For more information on how to use this wrapper, see [the Apify integration documentation](https://docs.apify.com/platform/integrations/langchain).


## Document loader

You can also use our `ApifyDatasetLoader` to get data from Apify dataset.
"""
logger.info("## Document loader")


"""
For a more detailed walkthrough of this loader, see [this notebook](/docs/integrations/document_loaders/apify_dataset).


Source code for this integration can be found in the [LangChain Apify repository](https://github.com/apify/langchain-apify).
"""
logger.info("For a more detailed walkthrough of this loader, see [this notebook](/docs/integrations/document_loaders/apify_dataset).")

logger.info("\n\n[DONE]", bright=True)