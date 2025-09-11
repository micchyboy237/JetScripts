from jet.logger import logger
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
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
# DuckDuckGo Search

This guide shows over how to use the DuckDuckGo search component.

## Usage
"""
logger.info("# DuckDuckGo Search")

# %pip install -qU duckduckgo-search langchain-community


search = DuckDuckGoSearchRun()

search.invoke("Obama's first name?")

"""
To get more additional information (e.g. link, source) use `DuckDuckGoSearchResults()`
"""
logger.info("To get more additional information (e.g. link, source) use `DuckDuckGoSearchResults()`")


search = DuckDuckGoSearchResults()

search.invoke("Obama")

"""
By default the results are returned as a comma-separated string of key-value pairs from the original search results. You can also choose to return the search results as a list by setting `output_format="list"` or as a JSON string by setting `output_format="json"`.
"""
logger.info("By default the results are returned as a comma-separated string of key-value pairs from the original search results. You can also choose to return the search results as a list by setting `output_format="list"` or as a JSON string by setting `output_format="json"`.")

search = DuckDuckGoSearchResults(output_format="list")

search.invoke("Obama")

"""
You can also just search for news articles. Use the keyword `backend="news"`
"""
logger.info("You can also just search for news articles. Use the keyword `backend="news"`")

search = DuckDuckGoSearchResults(backend="news")

search.invoke("Obama")

"""
You can also directly pass a custom `DuckDuckGoSearchAPIWrapper` to `DuckDuckGoSearchResults` to provide more control over the search results.
"""
logger.info("You can also directly pass a custom `DuckDuckGoSearchAPIWrapper` to `DuckDuckGoSearchResults` to provide more control over the search results.")


wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)

search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

search.invoke("Obama")

"""
## Related

- [How to use a chat model to call tools](https://python.langchain.com/docs/how_to/tool_calling/)
"""
logger.info("## Related")

logger.info("\n\n[DONE]", bright=True)