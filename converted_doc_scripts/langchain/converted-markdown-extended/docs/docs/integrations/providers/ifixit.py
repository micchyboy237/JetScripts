from jet.logger import logger
from langchain_community.document_loaders import IFixitLoader
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
# iFixit

>[iFixit](https://www.ifixit.com) is the largest, open repair community on the web. The site contains nearly 100k
> repair manuals, 200k Questions & Answers on 42k devices, and all the data is licensed under `CC-BY-NC-SA 3.0`.

## Installation and Setup

There isn't any special setup for it.

## Document Loader

See a [usage example](/docs/integrations/document_loaders/ifixit).
"""
logger.info("# iFixit")


logger.info("\n\n[DONE]", bright=True)