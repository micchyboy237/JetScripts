from jet.logger import logger
from langchain_community.document_loaders import UnstructuredMarkdownLoader
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
# UnstructuredMarkdownLoader

This notebook provides a quick overview for getting started with UnstructuredMarkdown [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all __ModuleName__Loader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html).

## Overview
### Integration details


| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/unstructured/)|
| :--- | :--- | :---: | :---: |  :---: |
| [UnstructuredMarkdownLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ❌ | ❌ | ✅ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| UnstructuredMarkdownLoader | ✅ | ❌ | 

## Setup

To access UnstructuredMarkdownLoader document loader you'll need to install the `langchain-community` integration package and the `unstructured` python package.

### Credentials

No credentials are needed to use this loader.

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
logger.info("# UnstructuredMarkdownLoader")



"""
### Installation

Install **langchain_community** and **unstructured**
"""
logger.info("### Installation")

# %pip install -qU langchain_community unstructured

"""
## Initialization

Now we can instantiate our model object and load documents. 

You can run the loader in one of two modes: "single" and "elements". If you use "single" mode, the document will be returned as a single `Document` object. If you use "elements" mode, the unstructured library will split the document into elements such as `Title` and `NarrativeText`. You can pass in additional `unstructured` kwargs after mode to apply different `unstructured` settings.
"""
logger.info("## Initialization")


loader = UnstructuredMarkdownLoader(
    "./example_data/example.md",
    mode="single",
    strategy="fast",
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
## Load Elements

In this example we will load in the `elements` mode, which will return a list of the different elements in the markdown document:
"""
logger.info("## Load Elements")


loader = UnstructuredMarkdownLoader(
    "./example_data/example.md",
    mode="elements",
    strategy="fast",
)

docs = loader.load()
len(docs)

"""
As you see there are 29 elements that were pulled from the `example.md` file. The first element is the title of the document as expected:
"""
logger.info("As you see there are 29 elements that were pulled from the `example.md` file. The first element is the title of the document as expected:")

docs[0].page_content

"""
## API reference

For detailed documentation of all UnstructuredMarkdownLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)