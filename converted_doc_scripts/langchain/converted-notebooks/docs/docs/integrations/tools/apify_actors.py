from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_apify import ApifyActorsTool
from langchain_core.messages import ToolMessage
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
# Apify Actor

>[Apify Actors](https://docs.apify.com/platform/actors) are cloud programs designed for a wide range of web scraping, crawling, and data extraction tasks. These actors facilitate automated data gathering from the web, enabling users to extract, process, and store information efficiently. Actors can be used to perform tasks like scraping e-commerce sites for product details, monitoring price changes, or gathering search engine results. They integrate seamlessly with [Apify Datasets](https://docs.apify.com/platform/storage/dataset), allowing the structured data collected by actors to be stored, managed, and exported in formats like JSON, CSV, or Excel for further analysis or use.

## Overview

This notebook walks you through using [Apify Actors](https://docs.apify.com/platform/actors) with LangChain to automate web scraping and data extraction. The `langchain-apify` package integrates Apify's cloud-based tools with LangChain agents, enabling efficient data collection and processing for AI applications.

## Setup

This integration lives in the [langchain-apify](https://pypi.org/project/langchain-apify/) package. The package can be installed using pip.
"""
logger.info("# Apify Actor")

# %pip install langchain-apify

"""
### Prerequisites

- **Apify account**: Register your free Apify account [here](https://console.apify.com/sign-up).
- **Apify API token**: Learn how to get your API token in the [Apify documentation](https://docs.apify.com/platform/integrations/api).
"""
logger.info("### Prerequisites")


os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"
# os.environ["OPENAI_API_KEY"] = "your-ollama-api-key"

"""
## Instantiation

Here we instantiate the `ApifyActorsTool` to be able to call [RAG Web Browser](https://apify.com/apify/rag-web-browser) Apify Actor. This Actor provides web browsing functionality for AI and LLM applications, similar to the web browsing feature in ChatGPT. Any Actor from the [Apify Store](https://apify.com/store) can be used in this way.
"""
logger.info("## Instantiation")


tool = ApifyActorsTool("apify/rag-web-browser")

"""
## Invocation

The `ApifyActorsTool` takes a single argument, which is `run_input` - a dictionary that is passed as a run input to the Actor. Run input schema documentation can be found in the input section of the Actor details page. See [RAG Web Browser input schema](https://apify.com/apify/rag-web-browser/input-schema).
"""
logger.info("## Invocation")

tool.invoke({"run_input": {"query": "what is apify?", "maxResults": 2}})

"""
## Chaining

We can provide the created tool to an [agent](https://python.langchain.com/docs/tutorials/agents/). When asked to search for information, the agent will call the Apify Actor, which will search the web, and then retrieve the search results.
"""
logger.info("## Chaining")

# %pip install langgraph langchain-ollama


model = ChatOllama(model="llama3.2")
tools = [tool]
graph = create_react_agent(model, tools=tools)

inputs = {"messages": [("user", "search for what is Apify")]}
for s in graph.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, ToolMessage):
        continue
    message.pretty_logger.debug()

"""
## API reference

For more information on how to use this integration, see the [git repository](https://github.com/apify/langchain-apify) or the [Apify integration documentation](https://docs.apify.com/platform/integrations/langgraph).
"""
logger.info("## API reference")


logger.info("\n\n[DONE]", bright=True)