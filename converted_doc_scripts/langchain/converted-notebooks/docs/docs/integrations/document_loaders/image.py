from jet.logger import logger
from langchain_community.document_loaders.image import UnstructuredImageLoader
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
# Images

This covers how to load images into a document format that we can use downstream with other LangChain modules.

It uses [Unstructured](https://unstructured.io/) to handle a wide variety of image formats, such as `.jpg` and `.png`. Please see [this guide](/docs/integrations/providers/unstructured/) for more instructions on setting up Unstructured locally, including setting up required system dependencies.

## Using Unstructured
"""
logger.info("# Images")

# %pip install --upgrade --quiet "unstructured[all-docs]"


loader = UnstructuredImageLoader("./example_data/layout-parser-paper-screenshot.png")

data = loader.load()

data[0]

"""
### Retain Elements

Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can keep that separation by specifying `mode="elements"`.
"""
logger.info("### Retain Elements")

loader = UnstructuredImageLoader(
    "./example_data/layout-parser-paper-screenshot.png", mode="elements"
)

data = loader.load()

data[0]

logger.info("\n\n[DONE]", bright=True)