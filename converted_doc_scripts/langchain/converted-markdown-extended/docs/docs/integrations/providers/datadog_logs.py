from jet.logger import logger
from langchain_community.document_loaders import DatadogLogsLoader
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
# Datadog Logs

>[Datadog](https://www.datadoghq.com/) is a monitoring and analytics platform for cloud-scale applications.

## Installation and Setup
"""
logger.info("# Datadog Logs")

pip install datadog_api_client

"""
We must initialize the loader with the Datadog API key and APP key, and we need to set up the query to extract the desired logs.

## Document Loader

See a [usage example](/docs/integrations/document_loaders/datadog_logs).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)