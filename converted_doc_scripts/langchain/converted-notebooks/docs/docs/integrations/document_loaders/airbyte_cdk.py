from jet.logger import logger
from langchain_community.document_loaders.airbyte import AirbyteCDKLoader
from langchain_core.documents import Document
from source_github.source import SourceGithub  # plug in your own source here
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

# Airbyte CDK (Deprecated)

Note: `AirbyteCDKLoader` is deprecated. Please use [`AirbyteLoader`](/docs/integrations/document_loaders/airbyte) instead.

>[Airbyte](https://github.com/airbytehq/airbyte) is a data integration platform for ELT pipelines from APIs, databases & files to warehouses & lakes. It has the largest catalog of ELT connectors to data warehouses and databases.

A lot of source connectors are implemented using the [Airbyte CDK](https://docs.airbyte.com/connector-development/cdk-python/). This loader allows to run any of these connectors and return the data as documents.

## Installation

First, you need to install the `airbyte-cdk` python package.
"""
logger.info("# Airbyte CDK (Deprecated)")

# %pip install --upgrade --quiet  airbyte-cdk

"""
Then, either install an existing connector from the [Airbyte Github repository](https://github.com/airbytehq/airbyte/tree/master/airbyte-integrations/connectors) or create your own connector using the [Airbyte CDK](https://docs.airbyte.io/connector-development/connector-development).

For example, to install the Github connector, run
"""
logger.info("Then, either install an existing connector from the [Airbyte Github repository](https://github.com/airbytehq/airbyte/tree/master/airbyte-integrations/connectors) or create your own connector using the [Airbyte CDK](https://docs.airbyte.io/connector-development/connector-development).")

# %pip install --upgrade --quiet  "source_github@git+https://github.com/airbytehq/airbyte.git@master#subdirectory=airbyte-integrations/connectors/source-github"

"""
Some sources are also published as regular packages on PyPI

## Example

Now you can create an `AirbyteCDKLoader` based on the imported source. It takes a `config` object that's passed to the connector. You also have to pick the stream you want to retrieve records from by name (`stream_name`). Check the connectors documentation page and spec definition for more information on the config object and available streams. For the Github connectors these are:

* [https://github.com/airbytehq/airbyte/blob/master/airbyte-integrations/connectors/source-github/source_github/spec.json](https://github.com/airbytehq/airbyte/blob/master/airbyte-integrations/connectors/source-github/source_github/spec.json).
* [https://docs.airbyte.com/integrations/sources/github/](https://docs.airbyte.com/integrations/sources/github/)
"""
logger.info("## Example")


config = {
    "credentials": {"api_url": "api.github.com", "personal_access_token": "<token>"},
    "repository": "<repo>",
    "start_date": "<date from which to start retrieving records from in ISO format, e.g. 2020-10-20T00:00:00Z>",
}

issues_loader = AirbyteCDKLoader(
    source_class=SourceGithub, config=config, stream_name="issues"
)

"""
Now you can load documents the usual way
"""
logger.info("Now you can load documents the usual way")

docs = issues_loader.load()

"""
As `load` returns a list, it will block until all documents are loaded. To have better control over this process, you can also use the `lazy_load` method which returns an iterator instead:
"""
logger.info("As `load` returns a list, it will block until all documents are loaded. To have better control over this process, you can also use the `lazy_load` method which returns an iterator instead:")

docs_iterator = issues_loader.lazy_load()

"""
Keep in mind that by default the page content is empty and the metadata object contains all the information from the record. To create documents in a different way, pass in a record_handler function when creating the loader:
"""
logger.info("Keep in mind that by default the page content is empty and the metadata object contains all the information from the record. To create documents in a different way, pass in a record_handler function when creating the loader:")



def handle_record(record, id):
    return Document(
        page_content=record.data["title"] + "\n" + (record.data["body"] or ""),
        metadata=record.data,
    )


issues_loader = AirbyteCDKLoader(
    source_class=SourceGithub,
    config=config,
    stream_name="issues",
    record_handler=handle_record,
)

docs = issues_loader.load()

"""
## Incremental loads

Some streams allow incremental loading, this means the source keeps track of synced records and won't load them again. This is useful for sources that have a high volume of data and are updated frequently.

To take advantage of this, store the `last_state` property of the loader and pass it in when creating the loader again. This will ensure that only new records are loaded.
"""
logger.info("## Incremental loads")

last_state = issues_loader.last_state  # store safely

incremental_issue_loader = AirbyteCDKLoader(
    source_class=SourceGithub, config=config, stream_name="issues", state=last_state
)

new_docs = incremental_issue_loader.load()

logger.info("\n\n[DONE]", bright=True)