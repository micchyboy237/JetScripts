from google.cloud import bigtable
from google.cloud.bigtable.row_set import RowSet
from google.colab import auth
from jet.logger import logger
from langchain_core.documents import Document
from langchain_google_bigtable import BigtableLoader
from langchain_google_bigtable import BigtableSaver
from langchain_google_bigtable import Encoding
from langchain_google_bigtable import MetadataMapping
import google.cloud.bigtable.row_filters as row_filters
import json
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
# Google Bigtable

> [Bigtable](https://cloud.google.com/bigtable) is a key-value and wide-column store, ideal for fast access to structured, semi-structured, or unstructured data. Extend your database application to build AI-powered experiences leveraging Bigtable's Langchain integrations.

This notebook goes over how to use [Bigtable](https://cloud.google.com/bigtable) to [save, load and delete langchain documents](/docs/how_to#document-loaders) with `BigtableLoader` and `BigtableSaver`.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-bigtable-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-bigtable-python/blob/main/docs/document_loader.ipynb)

## Before You Begin

To run this notebook, you will need to do the following:

* [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
* [Enable the Bigtable API](https://console.cloud.google.com/flows/enableapi?apiid=bigtable.googleapis.com)
* [Create a Bigtable instance](https://cloud.google.com/bigtable/docs/creating-instance)
* [Create a Bigtable table](https://cloud.google.com/bigtable/docs/managing-tables)
* [Create Bigtable access credentials](https://developers.google.com/workspace/guides/create-credentials)

After confirmed access to database in the runtime environment of this notebook, filling the following values and run the cell before running example scripts.
"""
logger.info("# Google Bigtable")

INSTANCE_ID = "my_instance"  # @param {type:"string"}
TABLE_ID = "my_table"  # @param {type:"string"}

"""
### 🦜🔗 Library Installation

The integration lives in its own `langchain-google-bigtable` package, so we need to install it.
"""
logger.info("### 🦜🔗 Library Installation")

# %pip install -upgrade --quiet langchain-google-bigtable

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

### Using the saver

Save langchain documents with `BigtableSaver.add_documents(<documents>)`. To initialize `BigtableSaver` class you need to provide 2 things:

1. `instance_id` - An instance of Bigtable.
1. `table_id` - The name of the table within the Bigtable to store langchain documents.
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

saver = BigtableSaver(
    instance_id=INSTANCE_ID,
    table_id=TABLE_ID,
)

saver.add_documents(test_docs)

"""
### Querying for Documents from Bigtable
For more details on connecting to a Bigtable table, please check the [Python SDK documentation](https://cloud.google.com/python/docs/reference/bigtable/latest/client).

#### Load documents from table

Load langchain documents with `BigtableLoader.load()` or `BigtableLoader.lazy_load()`. `lazy_load` returns a generator that only queries database during the iteration. To initialize `BigtableLoader` class you need to provide:

1. `instance_id` - An instance of Bigtable.
1. `table_id` - The name of the table within the Bigtable to store langchain documents.
"""
logger.info("### Querying for Documents from Bigtable")


loader = BigtableLoader(
    instance_id=INSTANCE_ID,
    table_id=TABLE_ID,
)

for doc in loader.lazy_load():
    logger.debug(doc)
    break

"""
### Delete documents

Delete a list of langchain documents from Bigtable table with `BigtableSaver.delete(<documents>)`.
"""
logger.info("### Delete documents")


docs = loader.load()
logger.debug("Documents before delete: ", docs)

onedoc = test_docs[0]
saver.delete([onedoc])
logger.debug("Documents after delete: ", loader.load())

"""
## Advanced Usage

### Limiting the returned rows
There are two ways to limit the returned rows:

1. Using a [filter](https://cloud.google.com/python/docs/reference/bigtable/latest/row-filters)
2. Using a [row_set](https://cloud.google.com/python/docs/reference/bigtable/latest/row-set#google.cloud.bigtable.row_set.RowSet)
"""
logger.info("## Advanced Usage")


filter_loader = BigtableLoader(
    INSTANCE_ID, TABLE_ID, filter=row_filters.ColumnQualifierRegexFilter(b"os_build")
)



row_set = RowSet()
row_set.add_row_range_from_keys(
    start_key="phone#4c410523#20190501", end_key="phone#4c410523#201906201"
)

row_set_loader = BigtableLoader(
    INSTANCE_ID,
    TABLE_ID,
    row_set=row_set,
)

"""
### Custom client
The client created by default is the default client, using only admin=True option. To use a non-default, a [custom client](https://cloud.google.com/python/docs/reference/bigtable/latest/client#class-googlecloudbigtableclientclientprojectnone-credentialsnone-readonlyfalse-adminfalse-clientinfonone-clientoptionsnone-adminclientoptionsnone-channelnone) can be passed to the constructor.
"""
logger.info("### Custom client")


custom_client_loader = BigtableLoader(
    INSTANCE_ID,
    TABLE_ID,
    client=bigtable.Client(...),
)

"""
### Custom content
The BigtableLoader assumes there is a column family called `langchain`, that has a column called `content`, that contains values encoded in UTF-8. These defaults can be changed like so:
"""
logger.info("### Custom content")


custom_content_loader = BigtableLoader(
    INSTANCE_ID,
    TABLE_ID,
    content_encoding=Encoding.ASCII,
    content_column_family="my_content_family",
    content_column_name="my_content_column_name",
)

"""
### Metadata mapping
By default, the `metadata` map on the `Document` object will contain a single key, `rowkey`, with the value of the row's rowkey value. To add more items to that map, use metadata_mapping.
"""
logger.info("### Metadata mapping")



metadata_mapping_loader = BigtableLoader(
    INSTANCE_ID,
    TABLE_ID,
    metadata_mappings=[
        MetadataMapping(
            column_family="my_int_family",
            column_name="my_int_column",
            metadata_key="key_in_metadata_map",
            encoding=Encoding.INT_BIG_ENDIAN,
        ),
        MetadataMapping(
            column_family="my_custom_family",
            column_name="my_custom_column",
            metadata_key="custom_key",
            encoding=Encoding.CUSTOM,
            custom_decoding_func=lambda input: json.loads(input.decode()),
            custom_encoding_func=lambda input: str.encode(json.dumps(input)),
        ),
    ],
)

"""
### Metadata as JSON

If there is a column in Bigtable that contains a JSON string that you would like to have added to the output document metadata, it is possible to add the following parameters to BigtableLoader. Note, the default value for `metadata_as_json_encoding` is UTF-8.
"""
logger.info("### Metadata as JSON")

metadata_as_json_loader = BigtableLoader(
    INSTANCE_ID,
    TABLE_ID,
    metadata_as_json_encoding=Encoding.ASCII,
    metadata_as_json_family="my_metadata_as_json_family",
    metadata_as_json_name="my_metadata_as_json_column_name",
)

"""
### Customize BigtableSaver

The BigtableSaver is also customizable similar to BigtableLoader.
"""
logger.info("### Customize BigtableSaver")

saver = BigtableSaver(
    INSTANCE_ID,
    TABLE_ID,
    client=bigtable.Client(...),
    content_encoding=Encoding.ASCII,
    content_column_family="my_content_family",
    content_column_name="my_content_column_name",
    metadata_mappings=[
        MetadataMapping(
            column_family="my_int_family",
            column_name="my_int_column",
            metadata_key="key_in_metadata_map",
            encoding=Encoding.INT_BIG_ENDIAN,
        ),
        MetadataMapping(
            column_family="my_custom_family",
            column_name="my_custom_column",
            metadata_key="custom_key",
            encoding=Encoding.CUSTOM,
            custom_decoding_func=lambda input: json.loads(input.decode()),
            custom_encoding_func=lambda input: str.encode(json.dumps(input)),
        ),
    ],
    metadata_as_json_encoding=Encoding.ASCII,
    metadata_as_json_family="my_metadata_as_json_family",
    metadata_as_json_name="my_metadata_as_json_column_name",
)

logger.info("\n\n[DONE]", bright=True)