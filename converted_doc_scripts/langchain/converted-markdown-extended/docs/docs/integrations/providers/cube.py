from jet.logger import logger
from langchain_community.document_loaders import CubeSemanticLoader
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
# Cube

>[Cube](https://cube.dev/) is the Semantic Layer for building data apps. It helps
> data engineers and application developers access data from modern data stores,
> organize it into consistent definitions, and deliver it to every application.

## Installation and Setup

We have to get the API key and the URL of the Cube instance. See
[these instructions](https://cube.dev/docs/product/apis-integrations/rest-api#configuration-base-path).


## Document loader

### Cube Semantic Layer

See a [usage example](/docs/integrations/document_loaders/cube_semantic).
"""
logger.info("# Cube")


logger.info("\n\n[DONE]", bright=True)