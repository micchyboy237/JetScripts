from jet.logger import logger
from langchain_community.document_loaders import BSHTMLLoader
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
# BSHTMLLoader


This notebook provides a quick overview for getting started with BeautifulSoup4 [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all __ModuleName__Loader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.html_bs.BSHTMLLoader.html).


## Overview
### Integration details


| Class | Package | Local | Serializable | JS support|
| :--- | :--- | :---: | :---: |  :---: |
| [BSHTMLLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.html_bs.BSHTMLLoader.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| BSHTMLLoader | ✅ | ❌ | 

## Setup

To access BSHTMLLoader document loader you'll need to install the `langchain-community` integration package and the `bs4` python package.

### Credentials

No credentials are needed to use the `BSHTMLLoader` class.

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
logger.info("# BSHTMLLoader")



"""
### Installation

Install **langchain-community** and **bs4**.
"""
logger.info("### Installation")

# %pip install -qU langchain-community bs4

"""
## Initialization

Now we can instantiate our model object and load documents:

- TODO: Update model instantiation with relevant params.
"""
logger.info("## Initialization")


loader = BSHTMLLoader(
    file_path="./example_data/fake-content.html",
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
page[0]

"""
## Adding separator to BS4

We can also pass a separator to use when calling get_text on the soup
"""
logger.info("## Adding separator to BS4")

loader = BSHTMLLoader(
    file_path="./example_data/fake-content.html", get_text_separator=", "
)

docs = loader.load()
logger.debug(docs[0])

"""
## API reference

For detailed documentation of all BSHTMLLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.html_bs.BSHTMLLoader.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)