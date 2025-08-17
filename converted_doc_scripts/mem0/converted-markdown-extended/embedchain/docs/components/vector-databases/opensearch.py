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
title: OpenSearch
---

Install related dependencies using the following command:
"""
logger.info("title: OpenSearch")

pip install --upgrade 'embedchain[opensearch]'

"""
<CodeGroup>
"""


app = App.from_config(config_path="config.yaml")

"""

"""

vectordb:
  provider: opensearch
  config:
    collection_name: 'my-app'
    opensearch_url: 'https://localhost:9200'
    http_auth:
      - admin
      - admin
    vector_dimension: 1536
    use_ssl: false
    verify_certs: false

"""
</CodeGroup>

<Snippet file="missing-vector-db-tip.mdx" />
"""

logger.info("\n\n[DONE]", bright=True)