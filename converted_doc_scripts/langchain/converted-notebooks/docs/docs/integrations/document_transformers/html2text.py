from jet.logger import logger
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
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
# HTML to text

>[html2text](https://github.com/Alir3z4/html2text/) is a Python package that converts a page of `HTML` into clean, easy-to-read plain `ASCII text`. 

The ASCII also happens to be a valid `Markdown` (a text-to-HTML format).
"""
logger.info("# HTML to text")

# %pip install --upgrade --quiet html2text


urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()


urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

logger.debug(docs_transformed[0].page_content[1000:2000])

logger.debug(docs_transformed[1].page_content[1000:2000])

logger.info("\n\n[DONE]", bright=True)