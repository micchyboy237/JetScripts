from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.document_loaders import CloudBlobLoader
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_community.document_loaders.parsers import TesseractBlobParser
import os
import pprint
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
# PyPDFLoader

This notebook provides a quick overview for getting started with `PyPDF` [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all DocumentLoader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html).

  

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support|
| :--- | :--- | :---: | :---: |  :---: |
| [PyPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ |   
   
---------   

### Loader features

|   Source    | Document Lazy Loading | Native Async Support | Extract Images | Extract Tables |
|:-----------:| :---: | :---: | :---: |:---: |
| PyPDFLoader | ✅ | ❌ | ✅ | ❌  |

  

## Setup

### Credentials

No credentials are required to use `PyPDFLoader`.

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
logger.info("# PyPDFLoader")



"""
### Installation

Install **langchain-community** and **pypdf**.
"""
logger.info("### Installation")

# %pip install -qU langchain-community pypdf

"""
## Initialization

Now we can instantiate our model object and load documents:
"""
logger.info("## Initialization")


file_path = "./example_data/layout-parser-paper.pdf"
loader = PyPDFLoader(file_path)

"""
## Load
"""
logger.info("## Load")

docs = loader.load()
docs[0]


pprint.pp(docs[0].metadata)

"""
## Lazy Load
"""
logger.info("## Lazy Load")

pages = []
for doc in loader.lazy_load():
    pages.append(doc)
    if len(pages) >= 10:

        pages = []
len(pages)

logger.debug(pages[0].page_content[:100])
pprint.pp(pages[0].metadata)

"""
The metadata attribute contains at least the following keys:
- source
- page (if in mode *page*)
- total_page
- creationdate
- creator
- producer

Additional metadata are specific to each parser.
These pieces of information can be helpful (to categorize your PDFs for example).

#
#
 
S
p
l
i
t
t
i
n
g
 
m
o
d
e
 
&
 
c
u
s
t
o
m
 
p
a
g
e
s
 
d
e
l
i
m
i
t
e
r

When loading the PDF file you can split it in two different ways:
- By page
- As a single text flow

By default PyPDFLoader will split the PDF as a single text flow.

#
#
#
 
E
x
t
r
a
c
t
 
t
h
e
 
P
D
F
 
b
y
 
p
a
g
e
.
 
E
a
c
h
 
p
a
g
e
 
i
s
 
e
x
t
r
a
c
t
e
d
 
a
s
 
a
 
l
a
n
g
c
h
a
i
n
 
D
o
c
u
m
e
n
t
 
o
b
j
e
c
t
:
"""
logger.info("#")

loader = PyPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
)
docs = loader.load()
logger.debug(len(docs))
pprint.pp(docs[0].metadata)

"""
I
n
 
t
h
i
s
 
m
o
d
e
 
t
h
e
 
p
d
f
 
i
s
 
s
p
l
i
t
 
b
y
 
p
a
g
e
s
 
a
n
d
 
t
h
e
 
r
e
s
u
l
t
i
n
g
 
D
o
c
u
m
e
n
t
s
 
m
e
t
a
d
a
t
a
 
c
o
n
t
a
i
n
s
 
t
h
e
 
p
a
g
e
 
n
u
m
b
e
r
.
 
B
u
t
 
i
n
 
s
o
m
e
 
c
a
s
e
s
 
w
e
 
c
o
u
l
d
 
w
a
n
t
 
t
o
 
p
r
o
c
e
s
s
 
t
h
e
 
p
d
f
 
a
s
 
a
 
s
i
n
g
l
e
 
t
e
x
t
 
f
l
o
w
 
(
s
o
 
w
e
 
d
o
n
'
t
 
c
u
t
 
s
o
m
e
 
p
a
r
a
g
r
a
p
h
s
 
i
n
 
h
a
l
f
)
.
 
I
n
 
t
h
i
s
 
c
a
s
e
 
y
o
u
 
c
a
n
 
u
s
e
 
t
h
e
 
*
s
i
n
g
l
e
*
 
m
o
d
e
 
:

#
#
#
 
E
x
t
r
a
c
t
 
t
h
e
 
w
h
o
l
e
 
P
D
F
 
a
s
 
a
 
s
i
n
g
l
e
 
l
a
n
g
c
h
a
i
n
 
D
o
c
u
m
e
n
t
 
o
b
j
e
c
t
:
"""
logger.info("#")

loader = PyPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="single",
)
docs = loader.load()
logger.debug(len(docs))
pprint.pp(docs[0].metadata)

"""
L
o
g
i
c
a
l
l
y
,
 
i
n
 
t
h
i
s
 
m
o
d
e
,
 
t
h
e
 
‘
p
a
g
e
_
n
u
m
b
e
r
’
 
m
e
t
a
d
a
t
a
 
d
i
s
a
p
p
e
a
r
s
.
 
H
e
r
e
'
s
 
h
o
w
 
t
o
 
c
l
e
a
r
l
y
 
i
d
e
n
t
i
f
y
 
w
h
e
r
e
 
p
a
g
e
s
 
e
n
d
 
i
n
 
t
h
e
 
t
e
x
t
 
f
l
o
w
 
:

#
#
#
 
A
d
d
 
a
 
c
u
s
t
o
m
 
*
p
a
g
e
s
_
d
e
l
i
m
i
t
e
r
*
 
t
o
 
i
d
e
n
t
i
f
y
 
w
h
e
r
e
 
a
r
e
 
e
n
d
s
 
o
f
 
p
a
g
e
s
 
i
n
 
*
s
i
n
g
l
e
*
 
m
o
d
e
:
"""
logger.info("#")

loader = PyPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="single",
    pages_delimiter="\n-------THIS IS A CUSTOM END OF PAGE-------\n",
)
docs = loader.load()
logger.debug(docs[0].page_content[:5780])

"""
T
h
i
s
 
c
o
u
l
d
 
s
i
m
p
l
y
 
b
e
 
\
n
,
 
o
r
 
\
f
 
t
o
 
c
l
e
a
r
l
y
 
i
n
d
i
c
a
t
e
 
a
 
p
a
g
e
 
c
h
a
n
g
e
,
 
o
r
 
\
<
!
-
-
 
P
A
G
E
 
B
R
E
A
K
 
-
-
>
 
f
o
r
 
s
e
a
m
l
e
s
s
 
i
n
j
e
c
t
i
o
n
 
i
n
 
a
 
M
a
r
k
d
o
w
n
 
v
i
e
w
e
r
 
w
i
t
h
o
u
t
 
a
 
v
i
s
u
a
l
 
e
f
f
e
c
t
.

#
 
E
x
t
r
a
c
t
 
i
m
a
g
e
s
 
f
r
o
m
 
t
h
e
 
P
D
F

You can extract images from your PDFs with a choice of three different solutions:
- rapidOCR (lightweight Optical Character Recognition tool)
- Tesseract (OCR tool with high precision)
- Multimodal language model

You can tune these functions to choose the output format of the extracted images among *html*, *markdown* or *text*

The result is inserted between the last and the second-to-last paragraphs of text of the page.

#
#
#
 
E
x
t
r
a
c
t
 
i
m
a
g
e
s
 
f
r
o
m
 
t
h
e
 
P
D
F
 
w
i
t
h
 
r
a
p
i
d
O
C
R
:
"""
logger.info("#")

# %pip install -qU rapidocr-onnxruntime


loader = PyPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="markdown-img",
    images_parser=RapidOCRBlobParser(),
)
docs = loader.load()

logger.debug(docs[5].page_content)

"""
B
e
 
c
a
r
e
f
u
l
,
 
R
a
p
i
d
O
C
R
 
i
s
 
d
e
s
i
g
n
e
d
 
t
o
 
w
o
r
k
 
w
i
t
h
 
C
h
i
n
e
s
e
 
a
n
d
 
E
n
g
l
i
s
h
,
 
n
o
t
 
o
t
h
e
r
 
l
a
n
g
u
a
g
e
s
.

#
#
#
 
E
x
t
r
a
c
t
 
i
m
a
g
e
s
 
f
r
o
m
 
t
h
e
 
P
D
F
 
w
i
t
h
 
T
e
s
s
e
r
a
c
t
:
"""
logger.info("#")

# %pip install -qU pytesseract


loader = PyPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="html-img",
    images_parser=TesseractBlobParser(),
)
docs = loader.load()
logger.debug(docs[5].page_content)

"""
#
#
#
 
E
x
t
r
a
c
t
 
i
m
a
g
e
s
 
f
r
o
m
 
t
h
e
 
P
D
F
 
w
i
t
h
 
m
u
l
t
i
m
o
d
a
l
 
m
o
d
e
l
:
"""
logger.info("#")

# %pip install -qU langchain-ollama



load_dotenv()

# from getpass import getpass

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass("Ollama API key =")


loader = PyPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="markdown-img",
    images_parser=LLMImageBlobParser(model=ChatOllama(model="llama3.2")),
)
docs = loader.load()
logger.debug(docs[5].page_content)

"""
## Working with Files

Many document loaders involve parsing files. The difference between such loaders usually stems from how the file is parsed, rather than how the file is loaded. For example, you can use `open` to read the binary content of either a PDF or a markdown file, but you need different parsing logic to convert that binary data into text.

As a result, it can be helpful to decouple the parsing logic from the loading logic, which makes it easier to re-use a given parser regardless of how the data was loaded.
You can use this strategy to analyze different files, with the same parsing parameters.
"""
logger.info("## Working with Files")


loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path="./example_data/",
        glob="*.pdf",
    ),
    blob_parser=PyPDFParser(),
)
docs = loader.load()
logger.debug(docs[0].page_content)
pprint.pp(docs[0].metadata)

"""
I
t
 
i
s
 
p
o
s
s
i
b
l
e
 
t
o
 
w
o
r
k
 
w
i
t
h
 
f
i
l
e
s
 
f
r
o
m
 
c
l
o
u
d
 
s
t
o
r
a
g
e
.
"""
logger.info("I")


loader = GenericLoader(
    blob_loader=CloudBlobLoader(
        url="s3://mybucket",  # Supports s3://, az://, gs://, file:// schemes.
        glob="*.pdf",
    ),
    blob_parser=PyPDFParser(),
)
docs = loader.load()
logger.debug(docs[0].page_content)
pprint.pp(docs[0].metadata)

"""
## API reference

For detailed documentation of all `PyPDFLoader` features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)