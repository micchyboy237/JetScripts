from jet.logger import logger
from langchain_community.document_loaders import UnstructuredCHMLoader
from langchain_community.document_loaders import UnstructuredCSVLoader
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredFileIOLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredImageLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredODTLoader
from langchain_community.document_loaders import UnstructuredOrgModeLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredRSTLoader
from langchain_community.document_loaders import UnstructuredRTFLoader
from langchain_community.document_loaders import UnstructuredTSVLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_unstructured import UnstructuredLoader
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
# Unstructured

>The `unstructured` package from
[Unstructured.IO](https://www.unstructured.io/) extracts clean text from raw source documents like
PDFs and Word documents.
This page covers how to use the [`unstructured`](https://github.com/Unstructured-IO/unstructured)
ecosystem within LangChain.

## Installation and Setup

If you are using a loader that runs locally, use the following steps to get `unstructured` and its
dependencies running.

- For the smallest installation footprint and to take advantage of features not available in the
  open-source `unstructured` package, install the Python SDK with `pip install unstructured-client`
  along with `pip install langchain-unstructured` to use the `UnstructuredLoader` and partition
  remotely against the Unstructured API. This loader lives
  in a LangChain partner repo instead of the `langchain-community` repo and you will need an
  `api_key`, which you can generate a free key [here](https://unstructured.io/api-key/).
    - Unstructured's documentation for the sdk can be found here:
      https://docs.unstructured.io/api-reference/api-services/sdk

- To run everything locally, install the open-source python package with `pip install unstructured`
  along with `pip install langchain-community` and use the same `UnstructuredLoader` as mentioned above.
    - You can install document specific dependencies with extras, e.g. `pip install "unstructured[docx]"`. Learn more about extras [here](https://docs.unstructured.io/open-source/installation/full-installation).
    - To install the dependencies for all document types, use `pip install "unstructured[all-docs]"`.
- Install the following system dependencies if they are not already available on your system with e.g. `brew install` for Mac.
  Depending on what document types you're parsing, you may not need all of these.
    - `libmagic-dev` (filetype detection)
    - `poppler-utils` (images and PDFs)
    - `tesseract-ocr`(images and PDFs)
    - `qpdf` (PDFs)
    - `libreoffice` (MS Office docs)
    - `pandoc` (EPUBs)
- When running locally, Unstructured also recommends using Docker [by following this
  guide](https://docs.unstructured.io/open-source/installation/docker-installation) to ensure all
  system dependencies are installed correctly.

The Unstructured API requires API keys to make requests.
You can request an API key [here](https://unstructured.io/api-key-hosted) and start using it today!
Checkout the README [here](https://github.com/Unstructured-IO/unstructured-api) here to get started making API calls.
We'd love to hear your feedback, let us know how it goes in our [community slack](https://join.slack.com/t/unstructuredw-kbe4326/shared_invite/zt-1x7cgo0pg-PTptXWylzPQF9xZolzCnwQ).
And stay tuned for improvements to both quality and performance!
Check out the instructions
[here](https://github.com/Unstructured-IO/unstructured-api#dizzy-instructions-for-using-the-docker-image) if you'd like to self-host the Unstructured API or run it locally.


## Data Loaders

The primary usage of `Unstructured` is in data loaders.

### UnstructuredLoader

See a [usage example](/docs/integrations/document_loaders/unstructured_file) to see how you can use
this loader for both partitioning locally and remotely with the serverless Unstructured API.
"""
logger.info("# Unstructured")


"""
### UnstructuredCHMLoader

`CHM` means `Microsoft Compiled HTML Help`.
"""
logger.info("### UnstructuredCHMLoader")


"""
### UnstructuredCSVLoader

A `comma-separated values` (`CSV`) file is a delimited text file that uses
a comma to separate values. Each line of the file is a data record.
Each record consists of one or more fields, separated by commas.

See a [usage example](/docs/integrations/document_loaders/csv#unstructuredcsvloader).
"""
logger.info("### UnstructuredCSVLoader")


"""
### UnstructuredEmailLoader

See a [usage example](/docs/integrations/document_loaders/email).
"""
logger.info("### UnstructuredEmailLoader")


"""
### UnstructuredEPubLoader

[EPUB](https://en.wikipedia.org/wiki/EPUB) is an `e-book file format` that uses
the “.epub” file extension. The term is short for electronic publication and
is sometimes styled `ePub`. `EPUB` is supported by many e-readers, and compatible
software is available for most smartphones, tablets, and computers.

See a [usage example](/docs/integrations/document_loaders/epub).
"""
logger.info("### UnstructuredEPubLoader")


"""
### UnstructuredExcelLoader

See a [usage example](/docs/integrations/document_loaders/microsoft_excel).
"""
logger.info("### UnstructuredExcelLoader")


"""
### UnstructuredFileIOLoader

See a [usage example](/docs/integrations/document_loaders/google_drive#passing-in-optional-file-loaders).
"""
logger.info("### UnstructuredFileIOLoader")


"""
### UnstructuredHTMLLoader

See a [usage example](/docs/how_to/document_loader_html).
"""
logger.info("### UnstructuredHTMLLoader")


"""
### UnstructuredImageLoader

See a [usage example](/docs/integrations/document_loaders/image).
"""
logger.info("### UnstructuredImageLoader")


"""
### UnstructuredMarkdownLoader

See a [usage example](/docs/integrations/vectorstores/starrocks).
"""
logger.info("### UnstructuredMarkdownLoader")


"""
### UnstructuredODTLoader

The `Open Document Format for Office Applications (ODF)`, also known as `OpenDocument`,
is an open file format for word processing documents, spreadsheets, presentations
and graphics and using ZIP-compressed XML files. It was developed with the aim of
providing an open, XML-based file format specification for office applications.

See a [usage example](/docs/integrations/document_loaders/odt).
"""
logger.info("### UnstructuredODTLoader")


"""
### UnstructuredOrgModeLoader

An [Org Mode](https://en.wikipedia.org/wiki/Org-mode) document is a document editing, formatting, and organizing mode, designed for notes, planning, and authoring within the free software text editor Emacs.

See a [usage example](/docs/integrations/document_loaders/org_mode).
"""
logger.info("### UnstructuredOrgModeLoader")


"""
### UnstructuredPDFLoader

See a [usage example](/docs/how_to/document_loader_pdf/#layout-analysis-and-extraction-of-text-from-images).
"""
logger.info("### UnstructuredPDFLoader")


"""
### UnstructuredPowerPointLoader

See a [usage example](/docs/integrations/document_loaders/microsoft_powerpoint).
"""
logger.info("### UnstructuredPowerPointLoader")


"""
### UnstructuredRSTLoader

A `reStructured Text` (`RST`) file is a file format for textual data
used primarily in the Python programming language community for technical documentation.

See a [usage example](/docs/integrations/document_loaders/rst).
"""
logger.info("### UnstructuredRSTLoader")


"""
### UnstructuredRTFLoader

See a usage example in the API documentation.
"""
logger.info("### UnstructuredRTFLoader")


"""
### UnstructuredTSVLoader

A `tab-separated values` (`TSV`) file is a simple, text-based file format for storing tabular data.
Records are separated by newlines, and values within a record are separated by tab characters.

See a [usage example](/docs/integrations/document_loaders/tsv).
"""
logger.info("### UnstructuredTSVLoader")


"""
### UnstructuredURLLoader

See a [usage example](/docs/integrations/document_loaders/url).
"""
logger.info("### UnstructuredURLLoader")


"""
### UnstructuredWordDocumentLoader

See a [usage example](/docs/integrations/document_loaders/microsoft_word#using-unstructured).
"""
logger.info("### UnstructuredWordDocumentLoader")


"""
### UnstructuredXMLLoader

See a [usage example](/docs/integrations/document_loaders/xml).
"""
logger.info("### UnstructuredXMLLoader")


logger.info("\n\n[DONE]", bright=True)