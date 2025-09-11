from jet.logger import logger
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
# Oxylabs

[Oxylabs](https://oxylabs.io/) is a market-leading web intelligence collection platform, driven by the highest business,
ethics, and compliance standards, enabling companies worldwide to unlock data-driven insights.

[langchain-oxylabs](https://pypi.org/project/langchain-oxylabs/) implements
tools enabling LLMs to interact with Oxylabs Web Scraper API.


## Installation and Setup
"""
logger.info("# Oxylabs")

pip install langchain-oxylabs

"""
## Tools

See details on available tools [here](/docs/integrations/tools/oxylabs/).
"""
logger.info("## Tools")

logger.info("\n\n[DONE]", bright=True)