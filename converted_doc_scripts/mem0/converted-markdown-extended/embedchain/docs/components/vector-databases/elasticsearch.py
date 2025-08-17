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
title: Elasticsearch
---

Install related dependencies using the following command:
"""
logger.info("title: Elasticsearch")

pip install --upgrade 'embedchain[elasticsearch]'

"""
<Note>
You can configure the Elasticsearch connection by providing either `es_url` or `cloud_id`. If you are using the Elasticsearch Service on Elastic Cloud, you can find the `cloud_id` on the [Elastic Cloud dashboard](https://cloud.elastic.co/deployments).
</Note>

You can authorize the connection to Elasticsearch by providing either `basic_auth`, `api_key`, or `bearer_auth`.

<CodeGroup>
"""
logger.info("You can configure the Elasticsearch connection by providing either `es_url` or `cloud_id`. If you are using the Elasticsearch Service on Elastic Cloud, you can find the `cloud_id` on the [Elastic Cloud dashboard](https://cloud.elastic.co/deployments).")


app = App.from_config(config_path="config.yaml")

"""

"""

vectordb:
  provider: elasticsearch
  config:
    collection_name: 'es-index'
    cloud_id: 'deployment-name:xxxx'
    basic_auth:
      - elastic
      - <your_password>
    verify_certs: false

"""
</CodeGroup>

<Snippet file="missing-vector-db-tip.mdx" />
"""

logger.info("\n\n[DONE]", bright=True)