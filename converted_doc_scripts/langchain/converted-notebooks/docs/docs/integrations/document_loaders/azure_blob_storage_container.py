from jet.logger import logger
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
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
# Azure Blob Storage Container

>[Azure Blob Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction) is Microsoft's object storage solution for the cloud. Blob Storage is optimized for storing massive amounts of unstructured data. Unstructured data is data that doesn't adhere to a particular data model or definition, such as text or binary data.

`Azure Blob Storage` is designed for:
- Serving images or documents directly to a browser.
- Storing files for distributed access.
- Streaming video and audio.
- Writing to log files.
- Storing data for backup and restore, disaster recovery, and archiving.
- Storing data for analysis by an on-premises or Azure-hosted service.

This notebook covers how to load document objects from a container on `Azure Blob Storage`.
"""
logger.info("# Azure Blob Storage Container")

# %pip install --upgrade --quiet  azure-storage-blob


loader = AzureBlobStorageContainerLoader(conn_str="<conn_str>", container="<container>")

loader.load()

"""
## Specifying a prefix
You can also specify a prefix for more fine-grained control over what files to load.
"""
logger.info("## Specifying a prefix")

loader = AzureBlobStorageContainerLoader(
    conn_str="<conn_str>", container="<container>", prefix="<prefix>"
)

loader.load()

logger.info("\n\n[DONE]", bright=True)