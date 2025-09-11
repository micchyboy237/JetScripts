from azure.ai.resources.client import AIClient
from azure.identity import DefaultAzureCredential
from jet.logger import logger
from langchain_community.document_loaders import AzureAIDataLoader
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
# Azure AI Data

>[Azure AI Foundry (formerly Azure AI Studio)](https://ai.azure.com/) provides the capability to upload data assets to cloud storage and register existing data assets from the following sources:
>
>- `Microsoft OneLake`
>- `Azure Blob Storage`
>- `Azure Data Lake gen 2`

The benefit of this approach over `AzureBlobStorageContainerLoader` and `AzureBlobStorageFileLoader` is that authentication is handled seamlessly to cloud storage. You can use either *identity-based* data access control to the data or *credential-based* (e.g. SAS token, account key). In the case of credential-based data access you do not need to specify secrets in your code or set up key vaults - the system handles that for you.

This notebook covers how to load document objects from a data asset in AI Studio.
"""
logger.info("# Azure AI Data")

# %pip install --upgrade --quiet azureml-fsspec azure-ai-generative


client = AIClient(
    credential=DefaultAzureCredential(),
    subscription_id="<subscription_id>",
    resource_group_name="<resource_group_name>",
    project_name="<project_name>",
)

data_asset = client.data.get(name="<data_asset_name>", label="latest")

loader = AzureAIDataLoader(url=data_asset.path)

loader.load()

"""
## Specifying a glob pattern
You can also specify a glob pattern for more fine-grained control over what files to load. In the example below, only files with a `pdf` extension will be loaded.
"""
logger.info("## Specifying a glob pattern")

loader = AzureAIDataLoader(url=data_asset.path, glob="*.pdf")

loader.load()

logger.info("\n\n[DONE]", bright=True)