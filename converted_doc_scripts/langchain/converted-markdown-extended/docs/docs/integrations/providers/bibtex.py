from jet.logger import logger
from langchain_community.document_loaders import BibtexLoader
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
# BibTeX

>[BibTeX](https://www.ctan.org/pkg/bibtex) is a file format and reference management system commonly used in conjunction with `LaTeX` typesetting. It serves as a way to organize and store bibliographic information for academic and research documents.

## Installation and Setup

We have to install the `bibtexparser` and `pymupdf` packages.
"""
logger.info("# BibTeX")

pip install bibtexparser pymupdf

"""
## Document loader

See a [usage example](/docs/integrations/document_loaders/bibtex).
"""
logger.info("## Document loader")


logger.info("\n\n[DONE]", bright=True)