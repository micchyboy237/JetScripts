from jet.logger import logger
from langchain_community.tools import MojeekSearch
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
# Mojeek Search

The following notebook will explain how to get results using Mojeek Search. Please visit [Mojeek Website](https://www.mojeek.com/services/search/web-search-api/) to obtain an API key.
"""
logger.info("# Mojeek Search")


  # obtained from Mojeek Website

search = MojeekSearch.config(api_key=api_key, search_kwargs={"t": 10})

"""
In `search_kwargs` you can add any search parameter that you can find on [Mojeek Documentation](https://www.mojeek.com/support/api/search/request_parameters.html)
"""
logger.info("In `search_kwargs` you can add any search parameter that you can find on [Mojeek Documentation](https://www.mojeek.com/support/api/search/request_parameters.html)")

search.run("mojeek")

logger.info("\n\n[DONE]", bright=True)