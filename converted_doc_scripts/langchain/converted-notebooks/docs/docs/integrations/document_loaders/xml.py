from jet.logger import logger
from langchain_community.document_loaders import UnstructuredXMLLoader
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
# UnstructuredXMLLoader

This notebook provides a quick overview for getting started with UnstructuredXMLLoader [document loader](https://python.langchain.com/docs/concepts/document_loaders). The `UnstructuredXMLLoader` is used to load `XML` files. The loader works with `.xml` files. The page content will be the text extracted from the XML tags.


## Overview
### Integration details


| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/unstructured/)|
| :--- | :--- | :---: | :---: |  :---: |
| [UnstructuredXMLLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.xml.UnstructuredXMLLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ✅ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| UnstructuredXMLLoader | ✅ | ❌ | 

## Setup

To access UnstructuredXMLLoader document loader you'll need to install the `langchain-community` integration package.

### Credentials

No credentials are needed to use the UnstructuredXMLLoader

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
logger.info("# UnstructuredXMLLoader")



"""
### Installation

Install **langchain_community**.
"""
logger.info("### Installation")

# %pip install -qU langchain_community

"""
## Initialization

Now we can instantiate our model object and load documents:
"""
logger.info("## Initialization")


loader = UnstructuredXMLLoader(
    "./example_data/factbook.xml",
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

page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:

        page = []

"""
## API reference

For detailed documentation of all __ModuleName__Loader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.xml.UnstructuredXMLLoader.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)