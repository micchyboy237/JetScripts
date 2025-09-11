from jet.logger import logger
from langchain_box.blob_loaders import BoxBlobLoader
from langchain_box.document_loaders import BoxLoader
from langchain_box.utilities import BoxMetadataQuery
from langchain_box.utilities import BoxSearchOptions, DocumentFiles, SearchTypeFilter
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
sidebar_label: Box
---

# BoxLoader and BoxBlobLoader


The `langchain-box` package provides two methods to index your files from Box: `BoxLoader` and `BoxBlobLoader`. `BoxLoader` allows you to ingest text representations of files that have a text representation in Box. The `BoxBlobLoader` allows you download the blob for any document or image file for processing with the blob parser of your choice.

This notebook details getting started with both of these. For detailed documentation of all BoxLoader features and configurations head to the API Reference pages for [BoxLoader](https://python.langchain.com/api_reference/box/document_loaders/langchain_box.document_loaders.box.BoxLoader.html) and [BoxBlobLoader](https://python.langchain.com/api_reference/box/document_loaders/langchain_box.blob_loaders.box.BoxBlobLoader.html).

## Overview

The `BoxLoader` class helps you get your unstructured content from Box in Langchain's `Document` format. You can do this with either a `List[str]` containing Box file IDs, or with a `str` containing a Box folder ID.

The `BoxBlobLoader` class helps you get your unstructured content from Box in Langchain's `Blob` format. You can do this with a `List[str]` containing Box file IDs, a `str` containing a Box folder ID, a search query, or a `BoxMetadataQuery`.

If getting files from a folder with folder ID, you can also set a `Bool` to tell the loader to get all sub-folders in that folder, as well.

:::info
A Box instance can contain Petabytes of files, and folders can contain millions of files. Be intentional when choosing what folders you choose to index. And we recommend never getting all files from folder 0 recursively. Folder ID 0 is your root folder.
:::

The `BoxLoader` will skip files without a text representation, while the `BoxBlobLoader` will return blobs for all document and image files.

### Integration details

| Class | Package | Local | Serializable | JS support|
| :--- | :--- | :---: | :---: |  :---: |
| [BoxLoader](https://python.langchain.com/api_reference/box/document_loaders/langchain_box.document_loaders.box.BoxLoader.html) | [langchain_box](https://python.langchain.com/api_reference/box/index.html) | ✅ | ❌ | ❌ |
| [BoxBlobLoader](https://python.langchain.com/api_reference/box/document_loaders/langchain_box.blob_loaders.box.BoxBlobLoader.html) | [langchain_box](https://python.langchain.com/api_reference/box/index.html) | ✅ | ❌ | ❌ |
### Loader features
| Source | Document Lazy Loading | Async Support
| :---: | :---: | :---: |
| BoxLoader | ✅ | ❌ |
| BoxBlobLoader | ✅ | ❌ |

## Setup

In order to use the Box package, you will need a few things:

* A Box account — If you are not a current Box customer or want to test outside of your production Box instance, you can use a [free developer account](https://account.box.com/signup/n/developer#ty9l3).
* [A Box app](https://developer.box.com/guides/getting-started/first-application/) — This is configured in the [developer console](https://account.box.com/developers/console), and for Box AI, must have the `Manage AI` scope enabled. Here you will also select your authentication method
* The app must be [enabled by the administrator](https://developer.box.com/guides/authorization/custom-app-approval/#manual-approval). For free developer accounts, this is whomever signed up for the account.

### Credentials

For these examples, we will use [token authentication](https://developer.box.com/guides/authentication/tokens/developer-tokens). This can be used with any [authentication method](https://developer.box.com/guides/authentication/). Just get the token with whatever methodology. If you want to learn more about how to use other authentication types with `langchain-box`, visit the [Box provider](/docs/integrations/providers/box) document.
"""
logger.info("# BoxLoader and BoxBlobLoader")

# import getpass

# box_developer_token = getpass.getpass("Enter your Box Developer Token: ")

"""
T
o
 
e
n
a
b
l
e
 
a
u
t
o
m
a
t
e
d
 
t
r
a
c
i
n
g
 
o
f
 
y
o
u
r
 
m
o
d
e
l
 
c
a
l
l
s
,
 
s
e
t
 
y
o
u
r
 
[
L
a
n
g
S
m
i
t
h
]
(
h
t
t
p
s
:
/
/
d
o
c
s
.
s
m
i
t
h
.
l
a
n
g
c
h
a
i
n
.
c
o
m
/
)
 
A
P
I
 
k
e
y
:
"""
logger.info("T")



"""
### Installation

Install **langchain_box**.
"""
logger.info("### Installation")

# %pip install -qU langchain_box

"""
## Initialization

### Load files

If you wish to load files, you must provide the `List` of file ids at instantiation time.

This requires 1 piece of information:

* **box_file_ids** (`List[str]`)- A list of Box file IDs.

#### BoxLoader
"""
logger.info("## Initialization")


box_file_ids = ["1514555423624", "1514553902288"]

loader = BoxLoader(
    box_developer_token=box_developer_token,
    box_file_ids=box_file_ids,
    character_limit=10000,  # Optional. Defaults to no limit
)

"""
#### BoxBlobLoader
"""
logger.info("#### BoxBlobLoader")


box_file_ids = ["1514555423624", "1514553902288"]

loader = BoxBlobLoader(
    box_developer_token=box_developer_token, box_file_ids=box_file_ids
)

"""
### Load from folder

If you wish to load files from a folder, you must provide a `str` with the Box folder ID at instantiation time.

This requires 1 piece of information:

* **box_folder_id** (`str`)- A string containing a Box folder ID.

#### BoxLoader
"""
logger.info("### Load from folder")


box_folder_id = "260932470532"

loader = BoxLoader(
    box_folder_id=box_folder_id,
    recursive=False,  # Optional. return entire tree, defaults to False
    character_limit=10000,  # Optional. Defaults to no limit
)

"""
#### BoxBlobLoader
"""
logger.info("#### BoxBlobLoader")


box_folder_id = "260932470532"

loader = BoxBlobLoader(
    box_folder_id=box_folder_id,
    recursive=False,  # Optional. return entire tree, defaults to False
)

"""
### Search for files with BoxBlobLoader

If you need to search for files, the `BoxBlobLoader` offers two methods. First you can perform a full text search with optional search options to narrow down that search.

This requires 1 piece of information:

* **query** (`str`)- A string containing the search query to perform.

You can also provide a `BoxSearchOptions` object to narrow down that search
* **box_search_options** (`BoxSearchOptions`)

#### BoxBlobLoader search
"""
logger.info("### Search for files with BoxBlobLoader")


box_folder_id = "260932470532"

box_search_options = BoxSearchOptions(
    ancestor_folder_ids=[box_folder_id],
    search_type_filter=[SearchTypeFilter.FILE_CONTENT],
    created_date_range=["2023-01-01T00:00:00-07:00", "2024-08-01T00:00:00-07:00,"],
    file_extensions=[DocumentFiles.DOCX, DocumentFiles.PDF],
    k=200,
    size_range=[1, 1000000],
    updated_data_range=None,
)

loader = BoxBlobLoader(
    box_developer_token=box_developer_token,
    query="Victor",
    box_search_options=box_search_options,
)

"""
You can also search for content based on Box Metadata. If your Box instance uses Metadata, you can search for any documents that have a specific Metadata Template attached that meet a certain criteria, like returning any invoices with a total greater than or equal to $500 that were created last quarter.

This requires 1 piece of information:

* **query** (`str`)- A string containing the search query to perform.

You can also provide a `BoxSearchOptions` object to narrow down that search
* **box_search_options** (`BoxSearchOptions`)

#### BoxBlobLoader Metadata query
"""
logger.info("#### BoxBlobLoader Metadata query")


query = BoxMetadataQuery(
    template_key="enterprise_1234.myTemplate",
    query="total >= :value",
    query_params={"value": 100},
    ancestor_folder_id="260932470532",
)

loader = BoxBlobLoader(box_metadata_query=query)

"""
## Load

#### BoxLoader
"""
logger.info("## Load")

docs = loader.load()
docs[0]

logger.debug(docs[0].metadata)

"""
#### BoxBlobLoader
"""
logger.info("#### BoxBlobLoader")

for blob in loader.yield_blobs():
    logger.debug(f"Blob({blob})")

"""
## Lazy Load

#### BoxLoader only
"""
logger.info("## Lazy Load")

page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:

        page = []

"""
## Extra fields

All Box connectors offer the ability to select additional fields from the Box `FileFull` object to return as custom LangChain metadata. Each object accepts an optional `List[str]` called `extra_fields` containing the json key from the return object, like `extra_fields=["shared_link"]`.

The connector will add this field to the list of fields the integration needs to function and then add the results to the metadata returned in the `Document` or `Blob`, like `"metadata" : { "source" : "source, "shared_link" : "shared_link" }`. If the field is unavailable for that file, it will be returned as an empty string, like `"shared_link" : ""`.
"""
logger.info("## Extra fields")



"""
## API reference

For detailed documentation of all BoxLoader features and configurations head to the [API reference](https://python.langchain.com/api_reference/box/document_loaders/langchain_box.document_loaders.box.BoxLoader.html)


## Help

If you have questions, you can check out our [developer documentation](https://developer.box.com) or reach out to use in our [developer community](https://community.box.com).


"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)