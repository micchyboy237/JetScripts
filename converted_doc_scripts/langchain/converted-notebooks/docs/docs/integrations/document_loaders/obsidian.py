from jet.logger import logger
from langchain_community.document_loaders import ObsidianLoader
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
# Obsidian

>[Obsidian](https://obsidian.md/) is a powerful and extensible knowledge base
that works on top of your local folder of plain text files.

This notebook covers how to load documents from an `Obsidian` database.

Since `Obsidian` is just stored on disk as a folder of Markdown files, the loader just takes a path to this directory.

`Obsidian` files also sometimes contain [metadata](https://help.obsidian.md/Editing+and+formatting/Metadata) which is a YAML block at the top of the file. These values will be added to the document's metadata. (`ObsidianLoader` can also be passed a `collect_metadata=False` argument to disable this behavior.)
"""
logger.info("# Obsidian")


loader = ObsidianLoader("<path-to-obsidian>")

docs = loader.load()

logger.info("\n\n[DONE]", bright=True)