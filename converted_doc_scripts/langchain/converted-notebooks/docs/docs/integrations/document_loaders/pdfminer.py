from bs4 import BeautifulSoup
from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.document_loaders import CloudBlobLoader
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_community.document_loaders.parsers import PDFMinerParser
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_core.documents import Document
import os
import pprint
import re
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
# PDFMinerLoader

This notebook provides a quick overview for getting started with `PDFMiner` [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all __ModuleName__Loader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PDFMinerLoader.html).

  

## Overview
### Integration details

| Class                                                                                                                                                | Package | Local | Serializable | JS support|
|:-----------------------------------------------------------------------------------------------------------------------------------------------------| :--- | :---: | :---: |  :---: |
| [PDFMinerLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PDFMinerLoader.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ |

---------   

### Loader features

|     Source     | Document Lazy Loading | Native Async Support | Extract Images | Extract Tables |
|:--------------:| :---: | :---: | :---: |:---: |
| PDFMinerLoader | ✅ | ❌ | ✅ | ✅ |

  

## Setup

### Credentials

No credentials are required to use PDFMinerLoader

I
f
 
y
o
u
 
w
a
n
t
 
t
o
 
g
e
t
 
a
u
t
o
m
a
t
e
d
 
b
e
s
t
 
i
n
-
c
l
a
s
s
 
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
 
y
o
u
 
c
a
n
 
a
l
s
o
 
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
 
b
y
 
u
n
c
o
m
m
e
n
t
i
n
g
 
b
e
l
o
w
:
"""
logger.info("# PDFMinerLoader")



"""
### Installation

Install **langchain-community** and **pdfminer**.
"""
logger.info("### Installation")

# %pip install -qU langchain-community pdfminer.six

"""
## Initialization

Now we can instantiate our model object and load documents:
"""
logger.info("## Initialization")


file_path = "./example_data/layout-parser-paper.pdf"
loader = PDFMinerLoader(file_path)

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

## Splitting mode & custom pages delimiter

When loading the PDF file you can split it in two different ways:
- By page
- As a single text flow

By default PDFMinerLoader will split the PDF by page.

### Extract the PDF by page. Each page is extracted as a langchain Document object:
"""
logger.info("## Splitting mode & custom pages delimiter")

loader = PDFMinerLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
)
docs = loader.load()
logger.debug(len(docs))
pprint.pp(docs[0].metadata)

"""
In this mode the pdf is split by pages and the resulting Documents metadata contains the page number. But in some cases we could want to process the pdf as a single text flow (so we don't cut some paragraphs in half). In this case you can use the *single* mode :

### Extract the whole PDF as a single langchain Document object:
"""
logger.info("### Extract the whole PDF as a single langchain Document object:")

loader = PDFMinerLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="single",
)
docs = loader.load()
logger.debug(len(docs))
pprint.pp(docs[0].metadata)

"""
Logically, in this mode, the ‘page_number’ metadata disappears. Here's how to clearly identify where pages end in the text flow :

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

loader = PDFMinerLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="single",
    pages_delimiter="\n-------THIS IS A CUSTOM END OF PAGE-------\n",
)
docs = loader.load()
logger.debug(docs[0].page_content[:5780])

"""
This could simply be \n, or \f to clearly indicate a page change, or \<!-- PAGE BREAK --> for seamless injection in a Markdown viewer without a visual effect.

# Extract images from the PDF

You can extract images from your PDFs with a choice of three different solutions:
- rapidOCR (lightweight Optical Character Recognition tool)
- Tesseract (OCR tool with high precision)
- Multimodal language model

You can tune these functions to choose the output format of the extracted images among *html*, *markdown* or *text*

The result is inserted between the last and the second-to-last paragraphs of text of the page.

### Extract images from the PDF with rapidOCR:
"""
logger.info("# Extract images from the PDF")

# %pip install -qU rapidocr-onnxruntime


loader = PDFMinerLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="markdown-img",
    images_parser=RapidOCRBlobParser(),
)
docs = loader.load()

logger.debug(docs[5].page_content)

"""
Be careful, RapidOCR is designed to work with Chinese and English, not other languages.

### Extract images from the PDF with Tesseract:
"""
logger.info("### Extract images from the PDF with Tesseract:")

# %pip install -qU pytesseract


loader = PDFMinerLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="html-img",
    images_parser=TesseractBlobParser(),
)
docs = loader.load()
logger.debug(docs[5].page_content)

"""
### Extract images from the PDF with multimodal model:
"""
logger.info("### Extract images from the PDF with multimodal model:")

# %pip install -qU langchain-ollama



load_dotenv()

# from getpass import getpass

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass("Ollama API key =")


loader = PDFMinerLoader(
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
logger.info("## Working with Files")


loader = GenericLoader(
    blob_loader=CloudBlobLoader(
        url="s3://mybucket",  # Supports s3://, az://, gs://, file:// schemes.
        glob="*.pdf",
    ),
    blob_parser=PDFMinerParser(),
)
docs = loader.load()
logger.debug(docs[0].page_content)
pprint.pp(docs[0].metadata)

"""
## Using PDFMiner to generate HTML text

This can be helpful for chunking texts semantically into sections as the output html content can be parsed via `BeautifulSoup` to get more structured and rich information about font size, page numbers, PDF headers/footers, etc.
"""
logger.info("## Using PDFMiner to generate HTML text")


file_path = "./example_data/layout-parser-paper.pdf"
loader = PDFMinerPDFasHTMLLoader(file_path)
docs = loader.load()
docs[0]


soup = BeautifulSoup(docs[0].page_content, "html.parser")
content = soup.find_all("div")


cur_fs = None
cur_text = ""
snippets = []  # first collect all snippets that have the same font size
for c in content:
    sp = c.find("span")
    if not sp:
        continue
    st = sp.get("style")
    if not st:
        continue
    fs = re.findall(r"font-size:(\d+)px", st)
    if not fs:
        continue
    fs = int(fs[0])
    if not cur_fs:
        cur_fs = fs
    if fs == cur_fs:
        cur_text += c.text
    else:
        snippets.append((cur_text, cur_fs))
        cur_fs = fs
        cur_text = c.text
snippets.append((cur_text, cur_fs))


cur_idx = -1
semantic_snippets = []
for s in snippets:
    if (
        not semantic_snippets
        or s[1] > semantic_snippets[cur_idx].metadata["heading_font"]
    ):
        metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
        metadata.update(docs[0].metadata)
        semantic_snippets.append(Document(page_content="", metadata=metadata))
        cur_idx += 1
        continue

    if (
        not semantic_snippets[cur_idx].metadata["content_font"]
        or s[1] <= semantic_snippets[cur_idx].metadata["content_font"]
    ):
        semantic_snippets[cur_idx].page_content += s[0]
        semantic_snippets[cur_idx].metadata["content_font"] = max(
            s[1], semantic_snippets[cur_idx].metadata["content_font"]
        )
        continue

    metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
    metadata.update(docs[0].metadata)
    semantic_snippets.append(Document(page_content="", metadata=metadata))
    cur_idx += 1

logger.debug(semantic_snippets[4])

"""
## API reference

For detailed documentation of all `PDFMinerLoader` features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PDFMinerLoader.html
"""
logger.info("## API reference")


loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path="./example_data/",
        glob="*.pdf",
    ),
    blob_parser=PDFMinerParser(),
)
docs = loader.load()
logger.debug(docs[0].page_content)
pprint.pp(docs[0].metadata)

logger.info("\n\n[DONE]", bright=True)