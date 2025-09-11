from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_community.document_loaders import SnowflakeLoader
from langchain_huggingface import HuggingFaceEmbeddings
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
# Snowflake

> [Snowflake](https://www.snowflake.com/) is a cloud-based data-warehousing platform
> that allows you to store and query large amounts of data.

This page covers how to use the `Snowflake` ecosystem within `LangChain`.

## Embedding models

Snowflake offers their open-weight `arctic` line of embedding models for free
on [Hugging Face](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5). The most recent model, snowflake-arctic-embed-m-v1.5 feature [matryoshka embedding](https://arxiv.org/abs/2205.13147) which allows for effective vector truncation.
You can use these models via the
[HuggingFaceEmbeddings](/docs/integrations/text_embedding/huggingfacehub) connector:
"""
logger.info("# Snowflake")

pip install langchain-community sentence-transformers

"""

"""


model = HuggingFaceEmbeddings(model_name="snowflake/arctic-embed-m-v1.5")

"""
## Document loader

You can use the [`SnowflakeLoader`](/docs/integrations/document_loaders/snowflake)
to load data from Snowflake:
"""
logger.info("## Document loader")


logger.info("\n\n[DONE]", bright=True)