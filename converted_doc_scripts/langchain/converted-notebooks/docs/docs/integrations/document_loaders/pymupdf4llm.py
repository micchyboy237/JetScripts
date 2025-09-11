from IPython.display import Markdown, display
from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_pymupdf4llm import PyMuPDF4LLMParser
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
---
sidebar_label: PyMuPDF4LLM
---

# PyMuPDF4LLMLoader

This notebook provides a quick overview for getting started with PyMuPDF4LLM [document loader](https://python.langchain.com/docs/concepts/#document-loaders). For detailed documentation of all PyMuPDF4LLMLoader features and configurations head to the [GitHub repository](https://github.com/lakinduboteju/langchain-pymupdf4llm).

## Overview

### Integration details

| Class | Package | Local | Serializable | JS support |
| :--- | :--- | :---: | :---: |  :---: |
| [PyMuPDF4LLMLoader](https://github.com/lakinduboteju/langchain-pymupdf4llm) | [langchain-pymupdf4llm](https://pypi.org/project/langchain-pymupdf4llm) | ✅ | ❌ | ❌ |

### Loader features

| Source | Document Lazy Loading | Native Async Support | Extract Images | Extract Tables |
| :---: | :---: | :---: | :---: | :---: |
| PyMuPDF4LLMLoader | ✅ | ❌ | ✅ | ✅ |

## Setup

To access PyMuPDF4LLM document loader you'll need to install the `langchain-pymupdf4llm` integration package.

### Credentials

No credentials are required to use PyMuPDF4LLMLoader.

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
logger.info("# PyMuPDF4LLMLoader")



"""
### Installation

Install **langchain-community** and **langchain-pymupdf4llm**.
"""
logger.info("### Installation")

# %pip install -qU langchain-community langchain-pymupdf4llm

"""
## Initialization

Now we can instantiate our model object and load documents:
"""
logger.info("## Initialization")


file_path = "./example_data/layout-parser-paper.pdf"
loader = PyMuPDF4LLMLoader(file_path)

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


part = pages[0].page_content[778:1189]
logger.debug(part)
display(Markdown(part))

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

## Splitting mode & custom pages delimiter

When loading the PDF file you can split it in two different ways:
- By page
- As a single text flow

By default PyMuPDF4LLMLoader will split the PDF by page.

### Extract the PDF by page. Each page is extracted as a langchain Document object:
"""
logger.info("## Splitting mode & custom pages delimiter")

loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
)
docs = loader.load()

logger.debug(len(docs))
pprint.pp(docs[0].metadata)

"""
In this mode the pdf is split by pages and the resulting Documents metadata contains the `page` (page number). But in some cases we could want to process the pdf as a single text flow (so we don't cut some paragraphs in half). In this case you can use the *single* mode :

### Extract the whole PDF as a single langchain Document object:
"""
logger.info("### Extract the whole PDF as a single langchain Document object:")

loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="single",
)
docs = loader.load()

logger.debug(len(docs))
pprint.pp(docs[0].metadata)

"""
Logically, in this mode, the `page` (page_number) metadata disappears. Here's how to clearly identify where pages end in the text flow :

### Add a custom *pages_delimiter* to identify where are ends of pages in *single* mode:
"""
logger.info("### Add a custom *pages_delimiter* to identify where are ends of pages in *single* mode:")

loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="single",
    pages_delimiter="\n-------THIS IS A CUSTOM END OF PAGE-------\n\n",
)
docs = loader.load()

part = docs[0].page_content[10663:11317]
logger.debug(part)
display(Markdown(part))

"""
The default `pages_delimiter` is \n-----\n\n.
But this could simply be \n, or \f to clearly indicate a page change, or \<!-- PAGE BREAK --> for seamless injection in a Markdown viewer without a visual effect.

# Extract images from the PDF

You can extract images from your PDFs (in text form) with a choice of three different solutions:
- rapidOCR (lightweight Optical Character Recognition tool)
- Tesseract (OCR tool with high precision)
- Multimodal language model

The result is inserted at the end of text of the page.

### Extract images from the PDF with rapidOCR:
"""
logger.info("# Extract images from the PDF")

# %pip install -qU rapidocr-onnxruntime pillow


loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    extract_images=True,
    images_parser=RapidOCRBlobParser(),
)
docs = loader.load()

part = docs[5].page_content[1863:]
logger.debug(part)
display(Markdown(part))

"""
Be careful, RapidOCR is designed to work with Chinese and English, not other languages.

### Extract images from the PDF with Tesseract:
"""
logger.info("### Extract images from the PDF with Tesseract:")

# %pip install -qU pytesseract


loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    extract_images=True,
    images_parser=TesseractBlobParser(),
)
docs = loader.load()

logger.debug(docs[5].page_content[1863:])

"""
### Extract images from the PDF with multimodal model:
"""
logger.info("### Extract images from the PDF with multimodal model:")

# %pip install -qU langchain-ollama



load_dotenv()

# from getpass import getpass

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass("Ollama API key =")


loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    extract_images=True,
    images_parser=LLMImageBlobParser(
        model=ChatOllama(model="llama3.2")
    ),
)
docs = loader.load()

logger.debug(docs[5].page_content[1863:])

"""
# Extract tables from the PDF

With PyMUPDF4LLM you can extract tables from your PDFs in *markdown* format :
"""
logger.info("# Extract tables from the PDF")

loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    table_strategy="lines",
)
docs = loader.load()

part = docs[4].page_content[3210:]
logger.debug(part)
display(Markdown(part))

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
    blob_parser=PyMuPDF4LLMParser(),
)
docs = loader.load()

part = docs[0].page_content[:562]
logger.debug(part)
display(Markdown(part))

"""
## API reference

For detailed documentation of all PyMuPDF4LLMLoader features and configurations head to the GitHub repository: https://github.com/lakinduboteju/langchain-pymupdf4llm
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)