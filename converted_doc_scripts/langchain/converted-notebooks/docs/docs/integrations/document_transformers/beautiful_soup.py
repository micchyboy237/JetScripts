from jet.logger import logger
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
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
# Beautiful Soup

>[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) is a Python package for parsing 
> HTML and XML documents (including having malformed markup, i.e. non-closed tags, so named after tag soup). 
> It creates a parse tree for parsed pages that can be used to extract data from HTML,[3] which 
> is useful for web scraping.

`Beautiful Soup` offers fine-grained control over HTML content, enabling specific tag extraction, removal, and content cleaning. 

It's suited for cases where you want to extract specific information and clean up the HTML content according to your needs.

For example, we can scrape text content within `<p>, <li>, <div>, and <a>` tags from the HTML content:

* `<p>`: The paragraph tag. It defines a paragraph in HTML and is used to group together related sentences and/or phrases.
 
* `<li>`: The list item tag. It is used within ordered (`<ol>`) and unordered (`<ul>`) lists to define individual items within the list.
 
* `<div>`: The division tag. It is a block-level element used to group other inline or block-level elements.
 
* `<a>`: The anchor tag. It is used to define hyperlinks.
"""
logger.info("# Beautiful Soup")


loader = AsyncChromiumLoader(["https://www.wsj.com"])
html = loader.load()

bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    html, tags_to_extract=["p", "li", "div", "a"]
)

docs_transformed[0].page_content[0:500]

logger.info("\n\n[DONE]", bright=True)