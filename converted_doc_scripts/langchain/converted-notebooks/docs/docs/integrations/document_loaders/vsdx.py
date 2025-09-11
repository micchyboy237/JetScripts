from jet.logger import logger
from langchain_community.document_loaders import VsdxLoader
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
# Vsdx

> A [visio file](https://fr.wikipedia.org/wiki/Microsoft_Visio) (with extension .vsdx) is associated with Microsoft Visio, a diagram creation software. It stores information about the structure, layout, and graphical elements of a diagram. This format facilitates the creation and sharing of visualizations in areas such as business, engineering, and computer science.

A Visio file can contain multiple pages. Some of them may serve as the background for others, and this can occur across multiple layers. This **loader** extracts the textual content from each page and its associated pages, enabling the extraction of all visible text from each page, similar to what an OCR algorithm would do.

**WARNING** : Only Visio files with the **.vsdx** extension are compatible with this loader. Files with extensions such as .vsd, ... are not compatible because they cannot be converted to compressed XML.
"""
logger.info("# Vsdx")


loader = VsdxLoader(file_path="./example_data/fake.vsdx")
documents = loader.load()

"""
**Display loaded documents**
"""

for i, doc in enumerate(documents):
    logger.debug(f"\n------ Page {doc.metadata['page']} ------")
    logger.debug(f"Title page : {doc.metadata['page_name']}")
    logger.debug(f"Source : {doc.metadata['source']}")
    logger.debug("\n==> CONTENT <== ")
    logger.debug(doc.page_content)

logger.info("\n\n[DONE]", bright=True)