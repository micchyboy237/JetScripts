from jet.logger import logger
from langchain_community.document_loaders import LakeFSLoader
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
# lakeFS

>[lakeFS](https://docs.lakefs.io/) provides scalable version control over
> the data lake, and uses Git-like semantics to create and access those versions.

## Installation and Setup

Get the `ENDPOINT`, `LAKEFS_ACCESS_KEY`, and `LAKEFS_SECRET_KEY`.
You can find installation instructions [here](https://docs.lakefs.io/quickstart/launch.html).


## Document Loader

See a [usage example](/docs/integrations/document_loaders/lakefs).
"""
logger.info("# lakeFS")


logger.info("\n\n[DONE]", bright=True)