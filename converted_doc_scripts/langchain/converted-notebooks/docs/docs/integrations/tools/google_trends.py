from jet.logger import logger
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
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
# Google Trends

This notebook goes over how to use the Google Trends Tool to fetch trends information.

First, you need to sign up for an `SerpApi key` key at: https://serpapi.com/users/sign_up.

Then you must install `google-search-results` with the command:

`pip install google-search-results`

Then you will need to set the environment variable `SERPAPI_API_KEY` to your `SerpApi key`

[Alternatively you can pass the key in as a argument to the wrapper `serp_`]

## Use the Tool
"""
logger.info("# Google Trends")

# %pip install --upgrade --quiet  google-search-results langchain_community



os.environ["SERPAPI_API_KEY"] = ""
tool = GoogleTrendsQueryRun(api_wrapper=GoogleTrendsAPIWrapper())

tool.run("Water")

logger.info("\n\n[DONE]", bright=True)