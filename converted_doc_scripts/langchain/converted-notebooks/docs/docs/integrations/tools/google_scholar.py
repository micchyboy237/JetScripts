from jet.logger import logger
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
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
# Google Scholar

This notebook goes through how to use Google Scholar Tool
"""
logger.info("# Google Scholar")

# %pip install --upgrade --quiet  google-search-results langchain-community



os.environ["SERP_API_KEY"] = ""
tool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())
tool.run("LLM Models")

logger.info("\n\n[DONE]", bright=True)