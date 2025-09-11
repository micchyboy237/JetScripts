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

This loader fetches the logs from your applications in Datadog using the `datadog_api_client` Python package. You must initialize the loader with your `Datadog API key` and `APP key`, and you need to pass in the query to extract the desired logs.
"""
logger.info("# Datadog Logs")


# %pip install --upgrade --quiet  datadog-api-client

DD_API_KEY = "..."
DD_APP_KEY = "..."

query = "service:agent status:error"

loader = DatadogLogsLoader(
    query=query,
    api_key=DD_API_KEY,
    app_key=DD_APP_KEY,
    from_time=1688732708951,  # Optional, timestamp in milliseconds
    to_time=1688736308951,  # Optional, timestamp in milliseconds
    limit=100,  # Optional, default is 100
)

documents = loader.load()
documents

logger.info("\n\n[DONE]", bright=True)