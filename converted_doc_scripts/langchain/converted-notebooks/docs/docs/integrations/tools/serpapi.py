from jet.logger import logger
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
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

This notebook goes over how to use the SerpAPI component to search the web.
"""
logger.info("# SerpAPI")


search = SerpAPIWrapper()

search.run("Obama's first name?")

"""
## Custom Parameters
You can also customize the SerpAPI wrapper with arbitrary parameters. For example, in the below example we will use `bing` instead of `google`.
"""
logger.info("## Custom Parameters")

params = {
    "engine": "bing",
    "gl": "us",
    "hl": "en",
}
search = SerpAPIWrapper(params=params)

search.run("Obama's first name?")


custom_tool = Tool(
    name="web search",
    description="Search the web for information",
    func=search.run,
)

logger.info("\n\n[DONE]", bright=True)