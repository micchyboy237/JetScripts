from jet.logger import logger
from langchain_valyu import ValyuRetriever
from langchain_valyu import ValyuSearchTool
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
# Valyu Deep Search
>[Valyu](https://www.valyu.network/) allows AI applications and agents to search the internet and proprietary data sources for relevant LLM ready information.

This notebook goes over how to use Valyu in LangChain.

First, get an Valyu API key and add it as an environment variable. Get $10 free credit  by [signing up here](https://platform.valyu.network/).

## Setup

The integration lives in the `langchain-valyu` package.
"""
logger.info("# Valyu Deep Search")

# %pip install -qU langchain-valyu

"""
In order to use the package, you will also need to set the `VALYU_API_KEY` environment variable to your Valyu API key.

## Context Retriever

You can use the [`ValyuContextRetriever`](https://pypi.org/project/langchain-valyu/) in a standard retrieval pipeline.
"""
logger.info("## Context Retriever")


valyu_

valyu_retriever = ValyuRetriever(
    k=5,
    search_type="all",
    relevance_threshold=0.5,
    max_price=20.0,
    start_date="2024-01-01",
    end_date="2024-12-31",
    valyu_api_key=valyu_api_key,
)

docs = valyu_retriever.invoke("What are the benefits of renewable energy?")

for doc in docs:
    logger.debug(doc.page_content)
    logger.debug(doc.metadata)

"""
## Context Search Tool

You can use the `ValyuSearchTool` for advanced search queries.
"""
logger.info("## Context Search Tool")


search_tool = ValyuSearchTool(valyu_)

search_results = search_tool._run(
    query="What are agentic search-enhanced large reasoning models?",
    search_type="all",
    max_num_results=5,
    relevance_threshold=0.5,
    max_price=20.0,
    start_date="2024-01-01",
    end_date="2024-12-31",
)

logger.debug("Search Results:", search_results)

logger.info("\n\n[DONE]", bright=True)