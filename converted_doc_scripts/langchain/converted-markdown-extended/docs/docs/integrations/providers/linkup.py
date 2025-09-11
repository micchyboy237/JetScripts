from jet.logger import logger
from langchain_linkup import LinkupSearchRetriever
from langchain_linkup import LinkupSearchTool
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
# Linkup

> [Linkup](https://www.linkup.so/) provides an API to connect LLMs to the web and the Linkup Premium Partner sources.

## Installation and Setup

To use the Linkup provider, you first need a valid API key, which you can find by signing-up [here](https://app.linkup.so/sign-up).
You will also need the `langchain-linkup` package, which you can install using pip:
"""
logger.info("# Linkup")

pip install langchain-linkup

"""
## Retriever

See a [usage example](/docs/integrations/retrievers/linkup_search).
"""
logger.info("## Retriever")


retriever = LinkupSearchRetriever(
    depth="deep",  # "standard" or "deep"
    linkup_api_key=None,  # API key can be passed here or set as the LINKUP_API_KEY environment variable
)

"""
## Tools

See a [usage example](/docs/integrations/tools/linkup_search).
"""
logger.info("## Tools")


tool = LinkupSearchTool(
    depth="deep",  # "standard" or "deep"
    output_type="searchResults",  # "searchResults", "sourcedAnswer" or "structured"
    linkup_api_key=None,  # API key can be passed here or set as the LINKUP_API_KEY environment variable
)

logger.info("\n\n[DONE]", bright=True)