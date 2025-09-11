from jet.logger import logger
from langchain_community.document_loaders import UnstructuredODTLoader
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
# Open Document Format (ODT)

>The [Open Document Format for Office Applications (ODF)](https://en.wikipedia.org/wiki/OpenDocument), also known as `OpenDocument`, is an open file format for word processing documents, spreadsheets, presentations and graphics and using ZIP-compressed XML files. It was developed with the aim of providing an open, XML-based file format specification for office applications.

>The standard is developed and maintained by a technical committee in the Organization for the Advancement of Structured Information Standards (`OASIS`) consortium. It was based on the Sun Microsystems specification for OpenOffice.org XML, the default format for `OpenOffice.org` and `LibreOffice`. It was originally developed for `StarOffice` "to provide an open standard for office documents."

The `UnstructuredODTLoader` is used to load `Open Office ODT` files.
"""
logger.info("# Open Document Format (ODT)")


loader = UnstructuredODTLoader("example_data/fake.odt", mode="elements")
docs = loader.load()
docs[0]

logger.info("\n\n[DONE]", bright=True)