from jet.logger import logger
from langchain_core.documents import Document
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
# Copy Paste

This notebook covers how to load a document object from something you just want to copy and paste. In this case, you don't even need to use a DocumentLoader, but rather can just construct the Document directly.
"""
logger.info("# Copy Paste")


text = "..... put the text you copy pasted here......"

doc = Document(page_content=text)

"""
## Metadata
If you want to add metadata about the where you got this piece of text, you easily can with the metadata key.
"""
logger.info("## Metadata")

metadata = {"source": "internet", "date": "Friday"}

doc = Document(page_content=text, metadata=metadata)

logger.info("\n\n[DONE]", bright=True)