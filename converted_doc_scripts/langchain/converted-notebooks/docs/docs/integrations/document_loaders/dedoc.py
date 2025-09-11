from jet.logger import logger
from langchain_community.document_loaders import DedocAPIFileLoader
from langchain_community.document_loaders import DedocFileLoader
from langchain_community.document_loaders import DedocPDFLoader
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
# Dedoc

This sample demonstrates the use of `Dedoc` in combination with `LangChain` as a `DocumentLoader`.

## Overview

[Dedoc](https://dedoc.readthedocs.io) is an [open-source](https://github.com/ispras/dedoc)
library/service that extracts texts, tables, attached files and document structure
(e.g., titles, list items, etc.) from files of various formats.

`Dedoc` supports `DOCX`, `XLSX`, `PPTX`, `EML`, `HTML`, `PDF`, images and more.
Full list of supported formats can be found [here](https://dedoc.readthedocs.io/en/latest/#id1).


### Integration details

| Class                                                                                                                                                | Package                                                                                        | Local | Serializable | JS support |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----:|:------------:|:----------:|
| [DedocFileLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.dedoc.DedocFileLoader.html)       | [langchain_community](https://python.langchain.com/api_reference/community/index.html) |   ❌   |     beta     |     ❌      |
| [DedocPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.DedocPDFLoader.html)           | [langchain_community](https://python.langchain.com/api_reference/community/index.html) |   ❌   |     beta     |     ❌      | 
| [DedocAPIFileLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.dedoc.DedocAPIFileLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) |   ❌   |     beta     |     ❌      | 


### Loader features

Methods for lazy loading and async loading are available, but in fact, document loading is executed synchronously.

|       Source       | Document Lazy Loading | Async Support |
|:------------------:|:---------------------:|:-------------:| 
|  DedocFileLoader   |           ❌           |       ❌       |
|   DedocPDFLoader   |           ❌           |       ❌       | 
| DedocAPIFileLoader |           ❌           |       ❌       | 

## Setup

* To access `DedocFileLoader` and `DedocPDFLoader` document loaders, you'll need to install the `dedoc` integration package.
* To access `DedocAPIFileLoader`, you'll need to run the `Dedoc` service, e.g. `Docker` container (please see [the documentation](https://dedoc.readthedocs.io/en/latest/getting_started/installation.html#install-and-run-dedoc-using-docker) 
for more details):

```bash
docker pull dedocproject/dedoc
docker run -p 1231:1231
```

`Dedoc` installation instruction is given [here](https://dedoc.readthedocs.io/en/latest/getting_started/installation.html).
"""
logger.info("# Dedoc")

# %pip install --quiet "dedoc[torch]"

"""
## Instantiation
"""
logger.info("## Instantiation")


loader = DedocFileLoader("./example_data/state_of_the_union.txt")

"""
## Load
"""
logger.info("## Load")

docs = loader.load()
docs[0].page_content[:100]

"""
## Lazy Load
"""
logger.info("## Lazy Load")

docs = loader.lazy_load()

for doc in docs:
    logger.debug(doc.page_content[:100])
    break

"""
## API reference

For detailed information on configuring and calling `Dedoc` loaders, please see the API references: 

* https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.dedoc.DedocFileLoader.html
* https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.DedocPDFLoader.html
* https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.dedoc.DedocAPIFileLoader.html

## Loading any file

For automatic handling of any file in a [supported format](https://dedoc.readthedocs.io/en/latest/#id1),
`DedocFileLoader` can be useful.
The file loader automatically detects the file type with a correct extension.

File parsing process can be configured through `dedoc_kwargs` during the `DedocFileLoader` class initialization.
Here the basic examples of some options usage are given, 
please see the documentation of `DedocFileLoader` and 
[dedoc documentation](https://dedoc.readthedocs.io/en/latest/parameters/parameters.html) 
to get more details about configuration parameters.

### Basic example
"""
logger.info("## API reference")


loader = DedocFileLoader("./example_data/state_of_the_union.txt")

docs = loader.load()

docs[0].page_content[:400]

"""
### Modes of split

`DedocFileLoader` supports different types of document splitting into parts (each part is returned separately).
For this purpose, `split` parameter is used with the following options:
* `document` (default value): document text is returned as a single langchain `Document` object (don't split);
* `page`: split document text into pages (works for `PDF`, `DJVU`, `PPTX`, `PPT`, `ODP`);
* `node`: split document text into `Dedoc` tree nodes (title nodes, list item nodes, raw text nodes);
* `line`: split document text into textual lines.
"""
logger.info("### Modes of split")

loader = DedocFileLoader(
    "./example_data/layout-parser-paper.pdf",
    split="page",
    pages=":2",
)

docs = loader.load()

len(docs)

"""
### Handling tables

`DedocFileLoader` supports tables handling when `with_tables` parameter is 
set to `True` during loader initialization (`with_tables=True` by default). 

Tables are not split - each table corresponds to one langchain `Document` object.
For tables, `Document` object has additional `metadata` fields `type="table"` 
and `text_as_html` with table `HTML` representation.
"""
logger.info("### Handling tables")

loader = DedocFileLoader("./example_data/mlb_teams_2012.csv")

docs = loader.load()

docs[1].metadata["type"], docs[1].metadata["text_as_html"][:200]

"""
### Handling attached files

`DedocFileLoader` supports attached files handling when `with_attachments` is set 
to `True` during loader initialization (`with_attachments=False` by default). 

Attachments are split according to the `split` parameter.
For attachments, langchain `Document` object has an additional metadata 
field `type="attachment"`.
"""
logger.info("### Handling attached files")

loader = DedocFileLoader(
    "./example_data/fake-email-attachment.eml",
    with_attachments=True,
)

docs = loader.load()

docs[1].metadata["type"], docs[1].page_content

"""
## Loading PDF file

If you want to handle only `PDF` documents, you can use `DedocPDFLoader` with only `PDF` support.
The loader supports the same parameters for document split, tables and attachments extraction.

`Dedoc` can extract `PDF` with or without a textual layer, 
as well as automatically detect its presence and correctness.
Several `PDF` handlers are available, you can use `pdf_with_text_layer` 
parameter to choose one of them.
Please see [parameters description](https://dedoc.readthedocs.io/en/latest/parameters/pdf_handling.html) 
to get more details.

For `PDF` without a textual layer, `Tesseract OCR` and its language packages should be installed.
In this case, [the instruction](https://dedoc.readthedocs.io/en/latest/tutorials/add_new_language.html) can be useful.
"""
logger.info("## Loading PDF file")


loader = DedocPDFLoader(
    "./example_data/layout-parser-paper.pdf", pdf_with_text_layer="true", pages="2:2"
)

docs = loader.load()

docs[0].page_content[:400]

"""
## Dedoc API

If you want to get up and running with less set up, you can use `Dedoc` as a service.
**`DedocAPIFileLoader` can be used without installation of `dedoc` library.**
The loader supports the same parameters as `DedocFileLoader` and
also automatically detects input file types.

To use `DedocAPIFileLoader`, you should run the `Dedoc` service, e.g. `Docker` container (please see [the documentation](https://dedoc.readthedocs.io/en/latest/getting_started/installation.html#install-and-run-dedoc-using-docker) 
for more details):

```bash
docker pull dedocproject/dedoc
docker run -p 1231:1231
```

Please do not use our demo URL `https://dedoc-readme.hf.space` in your code.
"""
logger.info("## Dedoc API")


loader = DedocAPIFileLoader(
    "./example_data/state_of_the_union.txt",
    url="https://dedoc-readme.hf.space",
)

docs = loader.load()

docs[0].page_content[:400]

logger.info("\n\n[DONE]", bright=True)