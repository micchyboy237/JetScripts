from jet.logger import logger
from langchain_community.embeddings import MosaicMLInstructorEmbeddings
import numpy as np
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
# MosaicML

>[MosaicML](https://docs.mosaicml.com/en/latest/inference.html) offers a managed inference service. You can either use a variety of open-source models, or deploy your own.

This example goes over how to use LangChain to interact with `MosaicML` Inference for text embedding.
"""
logger.info("# MosaicML")

# from getpass import getpass

# MOSAICML_API_TOKEN = getpass()


os.environ["MOSAICML_API_TOKEN"] = MOSAICML_API_TOKEN


embeddings = MosaicMLInstructorEmbeddings(
    query_instruction="Represent the query for retrieval: "
)

query_text = "This is a test query."
query_result = embeddings.embed_query(query_text)

document_text = "This is a test document."
document_result = embeddings.embed_documents([document_text])


query_numpy = np.array(query_result)
document_numpy = np.array(document_result[0])
similarity = np.dot(query_numpy, document_numpy) / (
    np.linalg.norm(query_numpy) * np.linalg.norm(document_numpy)
)
logger.debug(f"Cosine similarity between document and query: {similarity}")

logger.info("\n\n[DONE]", bright=True)