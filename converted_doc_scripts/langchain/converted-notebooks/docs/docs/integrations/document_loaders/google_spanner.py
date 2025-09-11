from google.cloud import spanner
from google.colab import auth
from google.oauth2 import service_account
from jet.logger import logger
from langchain_core.documents import Document
from langchain_google_spanner import Column
from langchain_google_spanner import SpannerDocumentSaver
from langchain_google_spanner import SpannerLoader
import datetime
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
# Google Spanner

> [Spanner](https://cloud.google.com/spanner) is a highly scalable database that combines unlimited scalability with relational semantics, such as secondary indexes, strong consistency, schemas, and SQL providing 99.999% availability in one easy solution.

This notebook goes over how to use [Spanner](https://cloud.google.com/spanner) to [save, load and delete langchain documents](/docs/how_to#document-loaders) with `SpannerLoader` and `SpannerDocumentSaver`.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-spanner-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-spanner-python/blob/main/docs/document_loader.ipynb)

## Before You Begin

To run this notebook, you will need to do the following:

* [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
* [Enable the Cloud Spanner API](https://console.cloud.google.com/flows/enableapi?apiid=spanner.googleapis.com)
* [Create a Spanner instance](https://cloud.google.com/spanner/docs/create-manage-instances)
* [Create a Spanner database](https://cloud.google.com/spanner/docs/create-manage-databases)
* [Create a Spanner table](https://cloud.google.com/spanner/docs/create-query-database-console#create-schema)

After confirmed access to database in the runtime environment of this notebook, filling the following values and run the cell before running example scripts.
"""
logger.info("# Google Spanner")

INSTANCE_ID = "test_instance"  # @param {type:"string"}
DATABASE_ID = "test_database"  # @param {type:"string"}
TABLE_NAME = "test_table"  # @param {type:"string"}

"""
### 🦜🔗 Library Installation

The integration lives in its own `langchain-google-spanner` package, so we need to install it.
"""
logger.info("### 🦜🔗 Library Installation")

# %pip install -upgrade --quiet langchain-google-spanner langchain

"""
**Colab only**: Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.
"""



"""
### ☁ Set Your Google Cloud Project
Set your Google Cloud project so that you can leverage Google Cloud resources within this notebook.

If you don't know your project ID, try the following:

* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113).
"""
logger.info("### ☁ Set Your Google Cloud Project")

PROJECT_ID = "my-project-id"  # @param {type:"string"}

# !gcloud config set project {PROJECT_ID}

"""
### 🔐 Authentication

Authenticate to Google Cloud as the IAM user logged into this notebook in order to access your Google Cloud Project.

- If you are using Colab to run this notebook, use the cell below and continue.
- If you are using Vertex AI Workbench, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).
"""
logger.info("### 🔐 Authentication")


auth.authenticate_user()

"""
## Basic Usage

### Save documents

Save langchain documents with `SpannerDocumentSaver.add_documents(<documents>)`. To initialize `SpannerDocumentSaver` class you need to provide 3 things:

1. `instance_id` - An instance of Spanner to load data from.
1. `database_id` - An instance of Spanner database to load data from.
1. `table_name` - The name of the table within the Spanner database to store langchain documents.
"""
logger.info("## Basic Usage")


test_docs = [
    Document(
        page_content="Apple Granny Smith 150 0.99 1",
        metadata={"fruit_id": 1},
    ),
    Document(
        page_content="Banana Cavendish 200 0.59 0",
        metadata={"fruit_id": 2},
    ),
    Document(
        page_content="Orange Navel 80 1.29 1",
        metadata={"fruit_id": 3},
    ),
]

saver = SpannerDocumentSaver(
    instance_id=INSTANCE_ID,
    database_id=DATABASE_ID,
    table_name=TABLE_NAME,
)
saver.add_documents(test_docs)

"""
### Querying for Documents from Spanner

For more details on connecting to a Spanner table, please check the [Python SDK documentation](https://cloud.google.com/python/docs/reference/spanner/latest).

#### Load documents from table

Load langchain documents with `SpannerLoader.load()` or `SpannerLoader.lazy_load()`. `lazy_load` returns a generator that only queries database during the iteration. To initialize `SpannerLoader` class you need to provide:

1. `instance_id` - An instance of Spanner to load data from.
1. `database_id` - An instance of Spanner database to load data from.
1. `query` - A query of the database dialect.
"""
logger.info("### Querying for Documents from Spanner")


query = f"SELECT * from {TABLE_NAME}"
loader = SpannerLoader(
    instance_id=INSTANCE_ID,
    database_id=DATABASE_ID,
    query=query,
)

for doc in loader.lazy_load():
    logger.debug(doc)
    break

"""
### Delete documents

Delete a list of langchain documents from the table with `SpannerDocumentSaver.delete(<documents>)`.
"""
logger.info("### Delete documents")

docs = loader.load()
logger.debug("Documents before delete:", docs)

doc = test_docs[0]
saver.delete([doc])
logger.debug("Documents after delete:", loader.load())

"""
## Advanced Usage

### Custom client

The client created by default is the default client. To pass in `credentials` and `project` explicitly, a custom client can be passed to the constructor.
"""
logger.info("## Advanced Usage")


creds = service_account.Credentials.from_service_account_file("/path/to/key.json")
custom_client = spanner.Client(project="my-project", credentials=creds)
loader = SpannerLoader(
    INSTANCE_ID,
    DATABASE_ID,
    query,
    client=custom_client,
)

"""
### Customize Document Page Content & Metadata

The loader will returns a list of Documents with page content from a specific data columns. All other data columns will be added to metadata. Each row becomes a document.

#### Customize page content format

The SpannerLoader assumes there is a column called `page_content`. These defaults can be changed like so:
"""
logger.info("### Customize Document Page Content & Metadata")

custom_content_loader = SpannerLoader(
    INSTANCE_ID, DATABASE_ID, query, content_columns=["custom_content"]
)

"""
If multiple columns are specified, the page content's string format will default to `text` (space-separated string concatenation). There are other format that user can specify, including `text`, `JSON`, `YAML`, `CSV`.

#### Customize metadata format

The SpannerLoader assumes there is a metadata column called `langchain_metadata` that store JSON data. The metadata column will be used as the base dictionary. By default, all other column data will be added and may overwrite the original value. These defaults can be changed like so:
"""
logger.info("#### Customize metadata format")

custom_metadata_loader = SpannerLoader(
    INSTANCE_ID, DATABASE_ID, query, metadata_columns=["column1", "column2"]
)

"""
#### Customize JSON metadata column name

By default, the loader uses `langchain_metadata` as the base dictionary. This can be customized to select a JSON column to use as base dictionary for the Document's metadata.
"""
logger.info("#### Customize JSON metadata column name")

custom_metadata_json_loader = SpannerLoader(
    INSTANCE_ID, DATABASE_ID, query, metadata_json_column="another-json-column"
)

"""
### Custom staleness

The default [staleness](https://cloud.google.com/python/docs/reference/spanner/latest/snapshot-usage#beginning-a-snapshot) is 15s. This can be customized by specifying a weaker bound (which can either be to perform all reads as of a given timestamp), or as of a given duration in the past.
"""
logger.info("### Custom staleness")


timestamp = datetime.datetime.utcnow()
custom_timestamp_loader = SpannerLoader(
    INSTANCE_ID,
    DATABASE_ID,
    query,
    staleness=timestamp,
)

duration = 20.0
custom_duration_loader = SpannerLoader(
    INSTANCE_ID,
    DATABASE_ID,
    query,
    staleness=duration,
)

"""
### Turn on data boost

By default, the loader will not use [data boost](https://cloud.google.com/spanner/docs/databoost/databoost-overview) since it has additional costs associated, and require additional IAM permissions. However, user can choose to turn it on.
"""
logger.info("### Turn on data boost")

custom_databoost_loader = SpannerLoader(
    INSTANCE_ID,
    DATABASE_ID,
    query,
    databoost=True,
)

"""
### Custom client

The client created by default is the default client. To pass in `credentials` and `project` explicitly, a custom client can be passed to the constructor.
"""
logger.info("### Custom client")


custom_client = spanner.Client(project="my-project", credentials=creds)
saver = SpannerDocumentSaver(
    INSTANCE_ID,
    DATABASE_ID,
    TABLE_NAME,
    client=custom_client,
)

"""
### Custom initialization for SpannerDocumentSaver

The SpannerDocumentSaver allows custom initialization. This allows user to specify how the Document is saved into the table.


content_column: This will be used as the column name for the Document's page content. Defaulted to `page_content`.

metadata_columns: These metadata will be saved into specific columns if the key exists in the Document's metadata.

metadata_json_column: This will be the column name for the spcial JSON column. Defaulted to `langchain_metadata`.
"""
logger.info("### Custom initialization for SpannerDocumentSaver")

custom_saver = SpannerDocumentSaver(
    INSTANCE_ID,
    DATABASE_ID,
    TABLE_NAME,
    content_column="my-content",
    metadata_columns=["foo"],
    metadata_json_column="my-special-json-column",
)

"""
### Initialize custom schema for Spanner

The SpannerDocumentSaver will have a `init_document_table` method to create a new table to store docs with custom schema.
"""
logger.info("### Initialize custom schema for Spanner")


new_table_name = "my_new_table"

SpannerDocumentSaver.init_document_table(
    INSTANCE_ID,
    DATABASE_ID,
    new_table_name,
    content_column="my-page-content",
    metadata_columns=[
        Column("category", "STRING(36)", True),
        Column("price", "FLOAT64", False),
    ],
)

logger.info("\n\n[DONE]", bright=True)