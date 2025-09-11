from jet.logger import logger
from langchain_community.document_loaders import UnstructuredEPubLoader
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
# EPub 

>[EPUB](https://en.wikipedia.org/wiki/EPUB) is an e-book file format that uses the ".epub" file extension. The term is short for electronic publication and is sometimes styled ePub. `EPUB` is supported by many e-readers, and compatible software is available for most smartphones, tablets, and computers.

This covers how to load `.epub` documents into the Document format that we can use downstream. You'll need to install the [`pandoc`](https://pandoc.org/installing.html) package for this loader to work with e.g. `brew install pandoc` for OSX.

Please see [this guide](/docs/integrations/providers/unstructured/) for more instructions on setting up Unstructured locally, including setting up required system dependencies.
"""
logger.info("# EPub")

# %pip install --upgrade --quiet unstructured


loader = UnstructuredEPubLoader("./example_data/childrens-literature.epub")

data = loader.load()

data[0]

"""
## Retain Elements

Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.
"""
logger.info("## Retain Elements")

loader = UnstructuredEPubLoader(
    "./example_data/childrens-literature.epub", mode="elements"
)

data = loader.load()

data[0]

logger.info("\n\n[DONE]", bright=True)