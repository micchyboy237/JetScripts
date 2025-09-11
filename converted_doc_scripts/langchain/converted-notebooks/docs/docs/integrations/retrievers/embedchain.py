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

>[Embedchain](https://github.com/embedchain/embedchain) is a RAG framework to create data pipelines. It loads, indexes, retrieves and syncs all the data.
>
>It is available as an [open source package](https://github.com/embedchain/embedchain) and as a [hosted platform solution](https://app.embedchain.ai/).

This notebook shows how to use a retriever that uses `Embedchain`.

# Installation

First you will need to install the [`embedchain` package](https://pypi.org/project/embedchain/). 

You can install the package by running
"""
logger.info("# Embedchain")

# %pip install --upgrade --quiet  embedchain

"""
# Create New Retriever

`EmbedchainRetriever` has a static `.create()` factory method that takes the following arguments:

* `yaml_path: string` optional -- Path to the YAML configuration file. If not provided, a default configuration is used. You can browse the [docs](https://docs.embedchain.ai/) to explore various customization options.
"""
logger.info("# Create New Retriever")

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()


retriever = EmbedchainRetriever.create()

"""
# Add Data

In embedchain, you can as many supported data types as possible. You can browse our [docs](https://docs.embedchain.ai/) to see the data types supported.

Embedchain automatically deduces the types of the data. So you can add a string, URL or local file path.
"""
logger.info("# Add Data")

retriever.add_texts(
    [
        "https://en.wikipedia.org/wiki/Elon_Musk",
        "https://www.forbes.com/profile/elon-musk",
        "https://www.youtube.com/watch?v=RcYjXbSJBN8",
    ]
)

"""
# Use Retriever

You can now use the retrieve to find relevant documents given a query
"""
logger.info("# Use Retriever")

result = retriever.invoke("How many companies does Elon Musk run and name those?")

result

logger.info("\n\n[DONE]", bright=True)