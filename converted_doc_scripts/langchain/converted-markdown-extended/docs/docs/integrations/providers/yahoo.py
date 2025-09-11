from jet.logger import logger
from langchain_community.tools import YahooFinanceNewsTool
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
# Yahoo

>[Yahoo (Wikipedia)](https://en.wikipedia.org/wiki/Yahoo) is an American web services provider.
>
> It provides a web portal, search engine Yahoo Search, and related
> services, including `My Yahoo`, `Yahoo Mail`, `Yahoo News`,
> `Yahoo Finance`, `Yahoo Sports` and its advertising platform, `Yahoo Native`.


## Tools

### Yahoo Finance News

We have to install a python package:
"""
logger.info("# Yahoo")

pip install yfinance

"""
See a [usage example](/docs/integrations/tools/yahoo_finance_news).
"""
logger.info("See a [usage example](/docs/integrations/tools/yahoo_finance_news).")


logger.info("\n\n[DONE]", bright=True)