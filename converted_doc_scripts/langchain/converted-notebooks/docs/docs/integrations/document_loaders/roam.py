from jet.logger import logger
from langchain_community.document_loaders import RoamLoader
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
# Roam

>[ROAM](https://roamresearch.com/) is a note-taking tool for networked thought, designed to create a personal knowledge base.

This notebook covers how to load documents from a Roam database. This takes a lot of inspiration from the example repo [here](https://github.com/JimmyLv/roam-qa).

## 🧑 Instructions for ingesting your own dataset

Export your dataset from Roam Research. You can do this by clicking on the three dots in the upper right hand corner and then clicking `Export`.

When exporting, make sure to select the `Markdown & CSV` format option.

This will produce a `.zip` file in your Downloads folder. Move the `.zip` file into this repository.

Run the following command to unzip the zip file (replace the `Export...` with your own file name as needed).

```shell
unzip Roam-Export-1675782732639.zip -d Roam_DB
```
"""
logger.info("# Roam")


loader = RoamLoader("Roam_DB")

docs = loader.load()

logger.info("\n\n[DONE]", bright=True)