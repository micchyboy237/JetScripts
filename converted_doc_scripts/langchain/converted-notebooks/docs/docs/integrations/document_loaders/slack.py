from jet.logger import logger
from langchain_community.document_loaders import SlackDirectoryLoader
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
# Slack

>[Slack](https://slack.com/) is an instant messaging program.

This notebook covers how to load documents from a Zipfile generated from a `Slack` export.

In order to get this `Slack` export, follow these instructions:

## 🧑 Instructions for ingesting your own dataset

Export your Slack data. You can do this by going to your Workspace Management page and clicking the Import/Export option (\{your_slack_domain\}.slack.com/services/export). Then, choose the right date range and click `Start export`. Slack will send you an email and a DM when the export is ready.

The download will produce a `.zip` file in your Downloads folder (or wherever your downloads can be found, depending on your OS configuration).

Copy the path to the `.zip` file, and assign it as `LOCAL_ZIPFILE` below.
"""
logger.info("# Slack")


SLACK_WORKSPACE_URL = "https://xxx.slack.com"
LOCAL_ZIPFILE = ""  # Paste the local path to your Slack zip file here.

loader = SlackDirectoryLoader(LOCAL_ZIPFILE, SLACK_WORKSPACE_URL)

docs = loader.load()
docs

logger.info("\n\n[DONE]", bright=True)