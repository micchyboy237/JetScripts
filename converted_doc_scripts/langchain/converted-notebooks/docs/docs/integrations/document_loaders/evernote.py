from jet.logger import logger
from langchain_community.document_loaders import EverNoteLoader
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
# EverNote

>[EverNote](https://evernote.com/) is intended for archiving and creating notes in which photos, audio and saved web content can be embedded. Notes are stored in virtual "notebooks" and can be tagged, annotated, edited, searched, and exported.

This notebook shows how to load an `Evernote` [export](https://help.evernote.com/hc/en-us/articles/209005557-Export-notes-and-notebooks-as-ENEX-or-HTML) file (.enex) from disk.

A document will be created for each note in the export.
"""
logger.info("# EverNote")

# %pip install --upgrade --quiet  lxml
# %pip install --upgrade --quiet  html2text


loader = EverNoteLoader("example_data/testing.enex")
loader.load()

loader = EverNoteLoader("example_data/testing.enex", load_single_document=False)
loader.load()

logger.info("\n\n[DONE]", bright=True)