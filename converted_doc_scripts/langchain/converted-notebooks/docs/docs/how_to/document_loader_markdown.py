from jet.logger import logger
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
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
# How to load Markdown

[Markdown](https://en.wikipedia.org/wiki/Markdown) is a lightweight markup language for creating formatted text using a plain-text editor.

Here we cover how to load `Markdown` documents into LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document) objects that we can use downstream.

We will cover:

- Basic usage;
- Parsing of Markdown into elements such as titles, list items, and text.

LangChain implements an [UnstructuredMarkdownLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html) object which requires the [Unstructured](https://docs.unstructured.io/welcome/) package. First we install it:
"""
logger.info("# How to load Markdown")

# %pip install "unstructured[md]" nltk

"""
Basic usage will ingest a Markdown file to a single document. Here we demonstrate on LangChain's readme:
"""
logger.info("Basic usage will ingest a Markdown file to a single document. Here we demonstrate on LangChain's readme:")


markdown_path = "../../../README.md"
loader = UnstructuredMarkdownLoader(markdown_path)

data = loader.load()
assert len(data) == 1
assert isinstance(data[0], Document)
readme_content = data[0].page_content
logger.debug(readme_content[:250])

"""
## Retain Elements

Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.
"""
logger.info("## Retain Elements")

loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")

data = loader.load()
logger.debug(f"Number of documents: {len(data)}\n")

for document in data[:2]:
    logger.debug(f"{document}\n")

"""
Note that in this case we recover three distinct element types:
"""
logger.info("Note that in this case we recover three distinct element types:")

logger.debug(set(document.metadata["category"] for document in data))

logger.info("\n\n[DONE]", bright=True)