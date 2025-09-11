from jet.logger import logger
from langchain_community.document_loaders import AsyncChromiumLoader
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
# Async Chromium

Chromium is one of the browsers supported by Playwright, a library used to control browser automation. 

By running `p.chromium.launch(headless=True)`, we are launching a headless instance of Chromium. 

Headless mode means that the browser is running without a graphical user interface.

In the below example we'll use the `AsyncChromiumLoader` to load the page, and then the [`Html2TextTransformer`](/docs/integrations/document_transformers/html2text/) to strip out the HTML tags and other semantic information.
"""
logger.info("# Async Chromium")

# %pip install --upgrade --quiet playwright beautifulsoup4 html2text
# !playwright install

"""
**Note:** If you are using Jupyter notebooks, you might also need to install and apply `nest_asyncio` before loading the documents like this:
"""

# !pip install nest-asyncio
# import nest_asyncio

# nest_asyncio.apply()


urls = ["https://docs.smith.langchain.com/"]
loader = AsyncChromiumLoader(urls, user_agent="MyAppUserAgent")
docs = loader.load()
docs[0].page_content[0:100]

"""
Now let's transform the documents into a more readable format using the transformer:
"""
logger.info("Now let's transform the documents into a more readable format using the transformer:")


html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
docs_transformed[0].page_content[0:500]

logger.info("\n\n[DONE]", bright=True)