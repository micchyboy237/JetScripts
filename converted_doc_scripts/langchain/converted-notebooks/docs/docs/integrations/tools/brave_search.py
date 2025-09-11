from jet.logger import logger
from langchain_community.tools import BraveSearch
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
# Brave Search

This notebook goes over how to use the Brave Search tool.
Go to the [Brave Website](https://brave.com/search/api/) to sign up for a free account and get an API key.
"""
logger.info("# Brave Search")

# %pip install --upgrade --quiet langchain-community




tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})

tool.run("obama middle name")

logger.info("\n\n[DONE]", bright=True)