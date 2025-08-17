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
title: Zilliz
---

Install related dependencies using the following command:
"""
logger.info("title: Zilliz")

pip install --upgrade 'embedchain[milvus]'

"""
Set the Zilliz environment variables `ZILLIZ_CLOUD_URI` and `ZILLIZ_CLOUD_TOKEN` which you can find it on their [cloud platform](https://cloud.zilliz.com/).

<CodeGroup>
"""
logger.info("Set the Zilliz environment variables `ZILLIZ_CLOUD_URI` and `ZILLIZ_CLOUD_TOKEN` which you can find it on their [cloud platform](https://cloud.zilliz.com/).")


os.environ['ZILLIZ_CLOUD_URI'] = 'https://xxx.zillizcloud.com'
os.environ['ZILLIZ_CLOUD_TOKEN'] = 'xxx'

app = App.from_config(config_path="config.yaml")

"""

"""

vectordb:
  provider: zilliz
  config:
    collection_name: 'zilliz_app'
    uri: https://xxxx.api.gcp-region.zillizcloud.com
    token: xxx
    vector_dim: 1536
    metric_type: L2

"""
</CodeGroup>

<Snippet file="missing-vector-db-tip.mdx" />
"""

logger.info("\n\n[DONE]", bright=True)