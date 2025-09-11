from jet.logger import logger
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
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
# UnstructuredPDFLoader

[Unstructured](https://unstructured-io.github.io/unstructured/) supports a common interface for working with unstructured or semi-structured file formats, such as Markdown or PDF. LangChain's [UnstructuredPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.UnstructuredPDFLoader.html) integrates with Unstructured to parse PDF documents into LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects.

Please see [this page](/docs/integrations/providers/unstructured/) for more information on installing system requirements.


### Integration details


| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/unstructured/)|
| :--- | :--- | :---: | :---: |  :---: |
| [UnstructuredPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.UnstructuredPDFLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ✅ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| UnstructuredPDFLoader | ✅ | ❌ | 

## Setup

### Credentials

No credentials are needed to use this loader.

To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("# UnstructuredPDFLoader")



"""
### Installation

Install **langchain_community** and **unstructured**.
"""
logger.info("### Installation")

# %pip install -qU langchain-community unstructured

"""
## Initialization

Now we can initialize our loader:
"""
logger.info("## Initialization")


file_path = "./example_data/layout-parser-paper.pdf"
loader = UnstructuredPDFLoader(file_path)

"""
## Load
"""
logger.info("## Load")

docs = loader.load()
docs[0]

logger.debug(docs[0].metadata)

"""
### Retain Elements

Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.
"""
logger.info("### Retain Elements")

file_path = "./example_data/layout-parser-paper.pdf"
loader = UnstructuredPDFLoader(file_path, mode="elements")

data = loader.load()
data[0]

"""
See the full set of element types for this particular document:
"""
logger.info("See the full set of element types for this particular document:")

set(doc.metadata["category"] for doc in data)

"""
### Fetching remote PDFs using Unstructured

This covers how to load online PDFs into a document format that we can use downstream. This can be used for various online PDF sites such as https://open.umn.edu/opentextbooks/textbooks/ and https://arxiv.org/archive/

Note: all other PDF loaders can also be used to fetch remote PDFs, but `OnlinePDFLoader` is a legacy function, and works specifically with `UnstructuredPDFLoader`.
"""
logger.info("### Fetching remote PDFs using Unstructured")


loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
data = loader.load()
data[0]

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

For detailed documentation of all UnstructuredPDFLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.UnstructuredPDFLoader.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)