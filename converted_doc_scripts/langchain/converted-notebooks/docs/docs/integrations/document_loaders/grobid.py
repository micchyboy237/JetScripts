from jet.logger import logger
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser
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
# Grobid

GROBID is a machine learning library for extracting, parsing, and re-structuring raw documents.

It is designed and expected to be used to parse academic papers, where it works particularly well. Note: if the articles supplied to Grobid are large documents (e.g. dissertations) exceeding a certain number of elements, they might not be processed. 

This loader uses Grobid to parse PDFs into `Documents` that retain metadata associated with the section of text.

---
The best approach is to install Grobid via docker, see https://grobid.readthedocs.io/en/latest/Grobid-docker/. 

(Note: additional instructions can be found [here](/docs/integrations/providers/grobid).)

Once grobid is up-and-running you can interact as described below.

Now, we can use the data loader.
"""
logger.info("# Grobid")


loader = GenericLoader.from_filesystem(
    "../Papers/",
    glob="*",
    suffixes=[".pdf"],
    parser=GrobidParser(segment_sentences=False),
)
docs = loader.load()

docs[3].page_content

docs[3].metadata

logger.info("\n\n[DONE]", bright=True)