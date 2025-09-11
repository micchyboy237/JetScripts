from jet.logger import logger
from langchain_community.document_loaders import JSONLoader
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
# JSONLoader

This notebook provides a quick overview for getting started with JSON [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all JSONLoader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.json_loader.JSONLoader.html).

- TODO: Add any other relevant links, like information about underlying API, etc.

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/json/)|
| :--- | :--- | :---: | :---: |  :---: |
| [JSONLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.json_loader.JSONLoader.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ✅ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| JSONLoader | ✅ | ❌ | 

## Setup

To access JSON document loader you'll need to install the `langchain-community` integration package as well as the ``jq`` python package.

### Credentials

No credentials are required to use the `JSONLoader` class.

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
logger.info("# JSONLoader")



"""
### Installation

Install **langchain-community** and **jq**:
"""
logger.info("### Installation")

# %pip install -qU langchain-community jq

"""
## Initialization

Now we can instantiate our model object and load documents:

- TODO: Update model instantiation with relevant params.
"""
logger.info("## Initialization")


loader = JSONLoader(
    file_path="./example_data/facebook_chat.json",
    jq_schema=".messages[].content",
    text_content=False,
)

"""
## Load
"""
logger.info("## Load")

docs = loader.load()
docs[0]

logger.debug(docs[0].metadata)

"""
## Lazy Load
"""
logger.info("## Lazy Load")

pages = []
for doc in loader.lazy_load():
    pages.append(doc)
    if len(pages) >= 10:

        pages = []

"""
## Read from JSON Lines file

If you want to load documents from a JSON Lines file, you pass `json_lines=True`
and specify `jq_schema` to extract `page_content` from a single JSON object.
"""
logger.info("## Read from JSON Lines file")

loader = JSONLoader(
    file_path="./example_data/facebook_chat_messages.jsonl",
    jq_schema=".content",
    text_content=False,
    json_lines=True,
)

docs = loader.load()
logger.debug(docs[0])

"""
## Read specific content keys

Another option is to set `jq_schema='.'` and provide a `content_key` in order to only load specific content:
"""
logger.info("## Read specific content keys")

loader = JSONLoader(
    file_path="./example_data/facebook_chat_messages.jsonl",
    jq_schema=".",
    content_key="sender_name",
    json_lines=True,
)

docs = loader.load()
logger.debug(docs[0])

"""
## JSON file with jq schema `content_key`

To load documents from a JSON file using the `content_key` within the jq schema, set `is_content_key_jq_parsable=True`. Ensure that `content_key` is compatible and can be parsed using the jq schema.
"""
logger.info("## JSON file with jq schema `content_key`")

loader = JSONLoader(
    file_path="./example_data/facebook_chat.json",
    jq_schema=".messages[]",
    content_key=".content",
    is_content_key_jq_parsable=True,
)

docs = loader.load()
logger.debug(docs[0])

"""
## Extracting metadata

Generally, we want to include metadata available in the JSON file into the documents that we create from the content.

The following demonstrates how metadata can be extracted using the `JSONLoader`.

There are some key changes to be noted. In the previous example where we didn't collect the metadata, we managed to directly specify in the schema where the value for the `page_content` can be extracted from.

In this example, we have to tell the loader to iterate over the records in the `messages` field. The jq_schema then has to be `.messages[]`

This allows us to pass the records (dict) into the `metadata_func` that has to be implemented. The `metadata_func` is responsible for identifying which pieces of information in the record should be included in the metadata stored in the final `Document` object.

Additionally, we now have to explicitly specify in the loader, via the `content_key` argument, the key from the record where the value for the `page_content` needs to be extracted from.
"""
logger.info("## Extracting metadata")

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["sender_name"] = record.get("sender_name")
    metadata["timestamp_ms"] = record.get("timestamp_ms")

    return metadata


loader = JSONLoader(
    file_path="./example_data/facebook_chat.json",
    jq_schema=".messages[]",
    content_key="content",
    metadata_func=metadata_func,
)

docs = loader.load()
logger.debug(docs[0].metadata)

"""
## API reference

For detailed documentation of all JSONLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.json_loader.JSONLoader.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)