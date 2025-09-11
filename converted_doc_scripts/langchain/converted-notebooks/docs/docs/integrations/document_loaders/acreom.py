from jet.logger import logger
from langchain_community.document_loaders import AcreomLoader
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
# acreom

[acreom](https://acreom.com) is a dev-first knowledge base with tasks running on local markdown files.

Below is an example on how to load a local acreom vault into Langchain. As the local vault in acreom is a folder of plain text .md files, the loader requires the path to the directory. 

Vault files may contain some metadata which is stored as a YAML header. These values will be added to the document’s metadata if `collect_metadata` is set to true.
"""
logger.info("# acreom")


loader = AcreomLoader("<path-to-acreom-vault>", collect_metadata=False)

docs = loader.load()

logger.info("\n\n[DONE]", bright=True)