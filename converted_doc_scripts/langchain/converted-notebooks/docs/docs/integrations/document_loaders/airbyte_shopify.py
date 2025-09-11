from jet.logger import logger
from langchain_community.document_loaders.airbyte import AirbyteShopifyLoader
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
---
sidebar_class_name: hidden
---

# Airbyte Shopify (Deprecated)

Note: This connector-specific loader is deprecated. Please use [`AirbyteLoader`](/docs/integrations/document_loaders/airbyte) instead.

>[Airbyte](https://github.com/airbytehq/airbyte) is a data integration platform for ELT pipelines from APIs, databases & files to warehouses & lakes. It has the largest catalog of ELT connectors to data warehouses and databases.

This loader exposes the Shopify connector as a document loader, allowing you to load various Shopify objects as documents.



## Installation

First, you need to install the `airbyte-source-shopify` python package.
"""
logger.info("# Airbyte Shopify (Deprecated)")

# %pip install --upgrade --quiet  airbyte-source-shopify

"""
## Example

Check out the [Airbyte documentation page](https://docs.airbyte.com/integrations/sources/shopify/) for details about how to configure the reader.
The JSON schema the config object should adhere to can be found on Github: [https://github.com/airbytehq/airbyte/blob/master/airbyte-integrations/connectors/source-shopify/source_shopify/spec.json](https://github.com/airbytehq/airbyte/blob/master/airbyte-integrations/connectors/source-shopify/source_shopify/spec.json).

The general shape looks like this:
```python
{
    "start_date": "<date from which to start retrieving records from in ISO format, e.g. 2020-10-20T00:00:00Z>",
    "shop": "<name of the shop you want to retrieve documents from>",
    "credentials": {
        "auth_method": "api_password",
        "api_password": "<your api password>"
    }
}
```

By default all fields are stored as metadata in the documents and the text is set to an empty string. Construct the text of the document by transforming the documents returned by the reader.
"""
logger.info("## Example")


config = {
}

loader = AirbyteShopifyLoader(
    config=config, stream_name="orders"
)  # check the documentation linked above for a list of all streams

"""
Now you can load documents the usual way
"""
logger.info("Now you can load documents the usual way")

docs = loader.load()

"""
As `load` returns a list, it will block until all documents are loaded. To have better control over this process, you can also use the `lazy_load` method which returns an iterator instead:
"""
logger.info("As `load` returns a list, it will block until all documents are loaded. To have better control over this process, you can also use the `lazy_load` method which returns an iterator instead:")

docs_iterator = loader.lazy_load()

"""
Keep in mind that by default the page content is empty and the metadata object contains all the information from the record. To create documents in a different way, pass in a record_handler function when creating the loader:
"""
logger.info("Keep in mind that by default the page content is empty and the metadata object contains all the information from the record. To create documents in a different way, pass in a record_handler function when creating the loader:")



def handle_record(record, id):
    return Document(page_content=record.data["title"], metadata=record.data)


loader = AirbyteShopifyLoader(
    config=config, record_handler=handle_record, stream_name="orders"
)
docs = loader.load()

"""
## Incremental loads

Some streams allow incremental loading, this means the source keeps track of synced records and won't load them again. This is useful for sources that have a high volume of data and are updated frequently.

To take advantage of this, store the `last_state` property of the loader and pass it in when creating the loader again. This will ensure that only new records are loaded.
"""
logger.info("## Incremental loads")

last_state = loader.last_state  # store safely

incremental_loader = AirbyteShopifyLoader(
    config=config, stream_name="orders", state=last_state
)

new_docs = incremental_loader.load()

logger.info("\n\n[DONE]", bright=True)