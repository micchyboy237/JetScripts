from jet.logger import logger
from langchain_community.retrievers import EmbedchainRetriever
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
# Embedchain

> [Embedchain](https://github.com/embedchain/embedchain) is a RAG framework to create
> data pipelines. It loads, indexes, retrieves and syncs all the data.
>
>It is available as an [open source package](https://github.com/embedchain/embedchain)
> and as a [hosted platform solution](https://app.embedchain.ai/).


## Installation and Setup

Install the package using pip:
"""
logger.info("# Embedchain")

pip install embedchain

"""
## Retriever

See a [usage example](/docs/integrations/retrievers/embedchain).
"""
logger.info("## Retriever")


logger.info("\n\n[DONE]", bright=True)