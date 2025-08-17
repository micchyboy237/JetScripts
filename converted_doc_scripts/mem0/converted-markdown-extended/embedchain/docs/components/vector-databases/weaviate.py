from embedchain import App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Weaviate
---


In order to use Weaviate as a vector database, set the environment variables `WEAVIATE_ENDPOINT` and `WEAVIATE_API_KEY` which you can find on [Weaviate dashboard](https://console.weaviate.cloud/dashboard).

<CodeGroup>
"""
logger.info("title: Weaviate")


app = App.from_config(config_path="config.yaml")

"""

"""

vectordb:
  provider: weaviate
  config:
    collection_name: my_weaviate_index

"""
</CodeGroup>

<Snippet file="missing-vector-db-tip.mdx" />
"""

logger.info("\n\n[DONE]", bright=True)