from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_community.document_transformers.nuclia_text_transform import (
NucliaTextTransformer,
)
from langchain_community.document_transformers.nuclia_text_transform import NucliaTextTransformer
from langchain_community.tools.nuclia import NucliaUnderstandingAPI
from langchain_core.documents import Document
import asyncio
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
# Nuclia

>[Nuclia](https://nuclia.com) automatically indexes your unstructured data from any internal and external source, providing optimized search results and generative answers. It can handle video and audio transcription, image content extraction, and document parsing.

`Nuclia Understanding API` document transformer splits text into paragraphs and sentences, identifies entities, provides a summary of the text and generates embeddings for all the sentences.

To use the Nuclia Understanding API, you need to have a Nuclia account. You can create one for free at [https://nuclia.cloud](https://nuclia.cloud), and then [create a NUA key](https://docs.nuclia.dev/docs/docs/using/understanding/intro).

"""
logger.info("# Nuclia")

# %pip install --upgrade --quiet  protobuf
# %pip install --upgrade --quiet  nucliadb-protos


os.environ["NUCLIA_ZONE"] = "<YOUR_ZONE>"  # e.g. europe-1
os.environ["NUCLIA_NUA_KEY"] = "<YOUR_API_KEY>"

"""
To use the Nuclia document transformer, you need to instantiate a `NucliaUnderstandingAPI` tool with `enable_ml` set to `True`:
"""
logger.info("To use the Nuclia document transformer, you need to instantiate a `NucliaUnderstandingAPI` tool with `enable_ml` set to `True`:")


nua = NucliaUnderstandingAPI(enable_ml=True)

"""
The Nuclia document transformer must be called in async mode, so you need to use the `atransform_documents` method:
"""
logger.info("The Nuclia document transformer must be called in async mode, so you need to use the `atransform_documents` method:")




async def process():
    documents = [
        Document(page_content="<TEXT 1>", metadata={}),
        Document(page_content="<TEXT 2>", metadata={}),
        Document(page_content="<TEXT 3>", metadata={}),
    ]
    nuclia_transformer = NucliaTextTransformer(nua)
    transformed_documents = await nuclia_transformer.atransform_documents(documents)
    logger.success(format_json(transformed_documents))
    logger.debug(transformed_documents)


asyncio.run(process())

logger.info("\n\n[DONE]", bright=True)