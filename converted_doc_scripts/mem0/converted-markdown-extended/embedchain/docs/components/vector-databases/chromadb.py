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
title: ChromaDB
---

<CodeGroup>
"""
logger.info("title: ChromaDB")


app = App.from_config(config_path="config1.yaml")

"""

"""

vectordb:
  provider: chroma
  config:
    collection_name: 'my-collection'
    dir: db
    allow_reset: true

"""

"""

vectordb:
  provider: chroma
  config:
    collection_name: 'my-collection'
    host: localhost
    port: 5200
    allow_reset: true

"""
</CodeGroup>

<Snippet file="missing-vector-db-tip.mdx" />
"""

logger.info("\n\n[DONE]", bright=True)