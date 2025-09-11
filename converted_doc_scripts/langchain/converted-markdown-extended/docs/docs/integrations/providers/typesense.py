from jet.logger import logger
from langchain_community.vectorstores import Typesense
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
# Typesense

> [Typesense](https://typesense.org) is an open-source, in-memory search engine, that you can either
> [self-host](https://typesense.org/docs/guide/install-typesense.html#option-2-local-machine-self-hosting) or run
> on [Typesense Cloud](https://cloud.typesense.org/).
> `Typesense` focuses on performance by storing the entire index in RAM (with a backup on disk) and also
> focuses on providing an out-of-the-box developer experience by simplifying available options and setting good defaults.

## Installation and Setup
"""
logger.info("# Typesense")

pip install typesense openapi-schema-pydantic

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/typesense).
"""
logger.info("## Vector Store")


logger.info("\n\n[DONE]", bright=True)