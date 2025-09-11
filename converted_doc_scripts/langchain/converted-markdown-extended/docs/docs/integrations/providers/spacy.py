from jet.logger import logger
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_text_splitters import SpacyTextSplitter
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
# spaCy

>[spaCy](https://spacy.io/) is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython.

## Installation and Setup
"""
logger.info("# spaCy")

pip install spacy

"""
## Text Splitter

See a [usage example](/docs/how_to/split_by_token/#spacy).
"""
logger.info("## Text Splitter")


"""
## Text Embedding Models

See a [usage example](/docs/integrations/text_embedding/spacy_embedding)
"""
logger.info("## Text Embedding Models")


logger.info("\n\n[DONE]", bright=True)