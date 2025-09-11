from jet.logger import logger
from langchain_community.document_loaders import __ModuleName__Loader
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
sidebar_label: __ModuleName__
---

# __ModuleName__Loader

- TODO: Make sure API reference link is correct.

This notebook provides a quick overview for getting started with __ModuleName__ [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all __ModuleName__Loader features and configurations head to the [API reference](https://python.langchain.com/v0.2/api_reference/community/document_loaders/langchain_community.document_loaders.__module_name___loader.__ModuleName__Loader.html).

- TODO: Add any other relevant links, like information about underlying API, etc.

## Overview
### Integration details

- TODO: Fill in table features.
- TODO: Remove JS support link if not relevant, otherwise ensure link is correct.
- TODO: Make sure API reference links are correct.

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/web_loaders/__module_name___loader)|
| :--- | :--- | :---: | :---: |  :---: |
| [__ModuleName__Loader](https://python.langchain.com/v0.2/api_reference/community/document_loaders/langchain_community.document_loaders.__module_name__loader.__ModuleName__Loader.html) | [langchain_community](https://api.python.langchain.com/en/latest/community_api_reference.html) | ✅/❌ | beta/❌ | ✅/❌ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| __ModuleName__Loader | ✅/❌ | ✅/❌ | 

## Setup

- TODO: Update with relevant info.

To access __ModuleName__ document loader you'll need to install the `__package_name__` integration package, and create a **ModuleName** account and get an API key.

### Credentials

- TODO: Update with relevant info.

Head to (TODO: link) to sign up to __ModuleName__ and generate an API key. Once you've done this set the __MODULE_NAME___API_KEY environment variable:
"""
logger.info("# __ModuleName__Loader")

# import getpass

# os.environ["__MODULE_NAME___API_KEY"] = getpass.getpass("Enter your __ModuleName__ API key: ")

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

Install **langchain_community**.

- TODO: Add any other required packages
"""
logger.info("### Installation")

# %pip install -qU langchain_community

"""
## Initialization

Now we can instantiate our model object and load documents:

- TODO: Update model instantiation with relevant params.
"""
logger.info("## Initialization")


loader = __ModuleName__Loader(
)

"""
## Load

- TODO: Run cells to show loading capabilities
"""
logger.info("## Load")

docs = loader.load()
docs[0]

logger.debug(docs[0].metadata)

"""
## Lazy Load

- TODO: Run cells to show lazy loading capabilities. Delete if lazy loading is not implemented.
"""
logger.info("## Lazy Load")

page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:

        page = []

"""
## TODO: Any functionality specific to this document loader

E.g. using specific configs for different loading behavior. Delete if not relevant.

## API reference

For detailed documentation of all __ModuleName__Loader features and configurations head to the API reference: https://python.langchain.com/v0.2/api_reference/community/document_loaders/langchain_community.document_loaders.__module_name___loader.__ModuleName__Loader.html
"""
logger.info("## TODO: Any functionality specific to this document loader")

logger.info("\n\n[DONE]", bright=True)