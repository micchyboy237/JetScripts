from jet.logger import logger
from langchain_community.document_loaders.nuclia import NucliaLoader
from langchain_community.tools.nuclia import NucliaUnderstandingAPI
import os
import shutil
import time


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
# Nuclia

>[Nuclia](https://nuclia.com) automatically indexes your unstructured data from any internal and external source, providing optimized search results and generative answers. It can handle video and audio transcription, image content extraction, and document parsing.

>The `Nuclia Understanding API` supports the processing of unstructured data, including text, web pages, documents, and audio/video contents. It extracts all texts wherever they are (using speech-to-text or OCR when needed), it also extracts metadata, embedded files (like images in a PDF), and web links. If machine learning is enabled, it identifies entities, provides a summary of the content and generates embeddings for all the sentences.

## Setup

To use the `Nuclia Understanding API`, you need to have a Nuclia account. You can create one for free at [https://nuclia.cloud](https://nuclia.cloud), and then [create a NUA key](https://docs.nuclia.dev/docs/docs/using/understanding/intro).
"""
logger.info("# Nuclia")

# %pip install --upgrade --quiet  protobuf
# %pip install --upgrade --quiet  nucliadb-protos


os.environ["NUCLIA_ZONE"] = "<YOUR_ZONE>"  # e.g. europe-1
os.environ["NUCLIA_NUA_KEY"] = "<YOUR_API_KEY>"

"""
## Example

To use the Nuclia document loader, you need to instantiate a `NucliaUnderstandingAPI` tool:
"""
logger.info("## Example")


nua = NucliaUnderstandingAPI(enable_ml=False)


loader = NucliaLoader("./interview.mp4", nua)

"""
You can now call the `load` the document in a loop until you get the document.
"""
logger.info("You can now call the `load` the document in a loop until you get the document.")


pending = True
while pending:
    time.sleep(15)
    docs = loader.load()
    if len(docs) > 0:
        logger.debug(docs[0].page_content)
        logger.debug(docs[0].metadata)
        pending = False
    else:
        logger.debug("waiting...")

"""
## Retrieved information

Nuclia returns the following information:

- file metadata
- extracted text
- nested text (like text in an embedded image)
- paragraphs and sentences splitting (defined by the position of their first and last characters, plus start time and end time for a video or audio file)
- links
- a thumbnail
- embedded files

Note:

  Generated files (thumbnail, extracted embedded files, etc.) are provided as a token. You can download them with the [`/processing/download` endpoint](https://docs.nuclia.dev/docs/api#operation/Download_binary_file_processing_download_get).

  Also at any level, if an attribute exceeds a certain size, it will be put in a downloadable file and will be replaced in the document by a file pointer. This will consist of `{"file": {"uri": "JWT_TOKEN"}}`. The rule is that if the size of the message is greater than 1000000 characters, the biggest parts will be moved to downloadable files. First, the compression process will target vectors. If that is not enough, it will target large field metadata, and finally it will target extracted text.
"""
logger.info("## Retrieved information")

logger.info("\n\n[DONE]", bright=True)