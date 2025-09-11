from docugami_langchain.document_loaders import DocugamiLoader
from jet.logger import logger
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
# Docugami

>[Docugami](https://docugami.com) converts business documents into a Document XML Knowledge Graph, generating forests
> of XML semantic trees representing entire documents. This is a rich representation that includes the semantic and
> structural characteristics of various chunks in the document as an XML tree.

## Installation and Setup
"""
logger.info("# Docugami")

pip install dgml-utils
pip install docugami-langchain

"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/docugami).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)