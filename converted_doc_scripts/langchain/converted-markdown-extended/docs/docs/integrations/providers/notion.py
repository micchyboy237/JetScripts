from jet.logger import logger
from langchain_community.document_loaders import NotionDirectoryLoader, NotionDBLoader
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
# Notion DB

>[Notion](https://www.notion.so/) is a collaboration platform with modified Markdown support that integrates kanban
> boards, tasks, wikis and databases. It is an all-in-one workspace for notetaking, knowledge and data management,
> and project and task management.

## Installation and Setup

All instructions are in examples below.

## Document Loader

We have two different loaders: `NotionDirectoryLoader` and `NotionDBLoader`.

See [usage examples here](/docs/integrations/document_loaders/notion).
"""
logger.info("# Notion DB")


logger.info("\n\n[DONE]", bright=True)