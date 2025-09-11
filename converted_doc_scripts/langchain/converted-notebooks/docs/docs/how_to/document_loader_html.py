from jet.logger import logger
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
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
# How to load HTML

The HyperText Markup Language or [HTML](https://en.wikipedia.org/wiki/HTML) is the standard markup language for documents designed to be displayed in a web browser.

This covers how to load `HTML` documents into a LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document) objects that we can use downstream.

Parsing HTML files often requires specialized tools. Here we demonstrate parsing via [Unstructured](https://docs.unstructured.io) and [BeautifulSoup4](https://beautiful-soup-4.readthedocs.io/en/latest/), which can be installed via pip. Head over to the integrations page to find integrations with additional services, such as [Azure AI Document Intelligence](/docs/integrations/document_loaders/azure_document_intelligence) or [FireCrawl](/docs/integrations/document_loaders/firecrawl).

## Loading HTML with Unstructured
"""
logger.info("# How to load HTML")

# %pip install unstructured


file_path = "../../docs/integrations/document_loaders/example_data/fake-content.html"

loader = UnstructuredHTMLLoader(file_path)
data = loader.load()

logger.debug(data)

"""
## Loading HTML with BeautifulSoup4

We can also use `BeautifulSoup4` to load HTML documents using the `BSHTMLLoader`.  This will extract the text from the HTML into `page_content`, and the page title as `title` into `metadata`.
"""
logger.info("## Loading HTML with BeautifulSoup4")

# %pip install bs4


loader = BSHTMLLoader(file_path)
data = loader.load()

logger.debug(data)

logger.info("\n\n[DONE]", bright=True)