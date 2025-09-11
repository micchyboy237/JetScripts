from jet.logger import logger
from langchain_community.retrievers import AskNewsRetriever
from langchain_community.tools.asknews import AskNewsSearch
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
# AskNews

[AskNews](https://asknews.app/) enhances language models with up-to-date global or historical news
by processing and indexing over 300,000 articles daily, providing prompt-optimized responses
through a low-latency endpoint, and ensuring transparency and diversity in its news coverage.

## Installation and Setup

First, you need to install `asknews` python package.
"""
logger.info("# AskNews")

pip install asknews

"""
You also need to set our AskNews API credentials, which can be generated at
the [AskNews console](https://my.asknews.app/).


## Retriever

See a [usage example](/docs/integrations/retrievers/asknews).
"""
logger.info("## Retriever")


"""
## Tool

See a [usage example](/docs/integrations/tools/asknews).
"""
logger.info("## Tool")


logger.info("\n\n[DONE]", bright=True)