from jet.logger import logger
from langchain_community.document_loaders.nuclia import NucliaLoader
from langchain_community.document_transformers.nuclia_text_transform import NucliaTextTransformer
from langchain_community.tools.nuclia import NucliaUnderstandingAPI
from langchain_community.vectorstores.nucliadb import NucliaDB
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
# Nuclia

>[Nuclia](https://nuclia.com) automatically indexes your unstructured data from any internal
> and external source, providing optimized search results and generative answers.
> It can handle video and audio transcription, image content extraction, and document parsing.



## Installation and Setup

We need to install the `nucliadb-protos` package to use the `Nuclia Understanding API`
"""
logger.info("# Nuclia")

pip install nucliadb-protos

"""
We need to have a `Nuclia account`.
We can create one for free at [https://nuclia.cloud](https://nuclia.cloud),
and then [create a NUA key](https://docs.nuclia.dev/docs/docs/using/understanding/intro).


## Document Transformer

### Nuclia

>`Nuclia Understanding API` document transformer splits text into paragraphs and sentences,
> identifies entities, provides a summary of the text and generates embeddings for all the sentences.

To use the Nuclia document transformer, we need to instantiate a `NucliaUnderstandingAPI`
tool with `enable_ml` set to `True`:
"""
logger.info("## Document Transformer")


nua = NucliaUnderstandingAPI(enable_ml=True)

"""
See a [usage example](/docs/integrations/document_transformers/nuclia_transformer).
"""
logger.info("See a [usage example](/docs/integrations/document_transformers/nuclia_transformer).")


"""
## Document Loaders

### Nuclea loader

See a [usage example](/docs/integrations/document_loaders/nuclia).
"""
logger.info("## Document Loaders")


"""
## Vector store

### NucliaDB

We need to install a python package:
"""
logger.info("## Vector store")

pip install nuclia

"""
See a [usage example](/docs/integrations/vectorstores/nucliadb).
"""
logger.info("See a [usage example](/docs/integrations/vectorstores/nucliadb).")


"""
## Tools

### Nuclia Understanding

See a [usage example](/docs/integrations/tools/nuclia).
"""
logger.info("## Tools")


logger.info("\n\n[DONE]", bright=True)