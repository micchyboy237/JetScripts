from jet.logger import logger
from langchain_fmp_data import FMPDataTool, FMPDataToolkit
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
# FMP Data (Financial Data Prep)

> [FMP-Data](https://pypi.org/project/fmp-data/) is a python package for connecting to
> Financial Data Prep API. It simplifies how you can access production quality data.


## Installation and Setup

Get an `FMP Data` API key by
visiting [this page](https://site.financialmodelingprep.com/pricing-plans?couponCode=mehdi).
 and set it as an environment variable (`FMP_API_KEY`).

Then, install [langchain-fmp-data](https://pypi.org/project/langchain-fmp-data/).

## Tools

See an [example](https://github.com/MehdiZare/langchain-fmp-data/tree/main/docs).
"""
logger.info("# FMP Data (Financial Data Prep)")


logger.info("\n\n[DONE]", bright=True)