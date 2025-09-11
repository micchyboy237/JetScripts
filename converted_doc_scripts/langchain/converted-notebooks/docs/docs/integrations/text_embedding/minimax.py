from jet.logger import logger
from langchain_community.embeddings import MiniMaxEmbeddings
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
# MiniMax

[MiniMax](https://api.minimax.chat/document/guides/embeddings?id=6464722084cdc277dfaa966a) offers an embeddings service.

This example goes over how to use LangChain to interact with MiniMax Inference for text embedding.
"""
logger.info("# MiniMax")


os.environ["MINIMAX_GROUP_ID"] = "MINIMAX_GROUP_ID"
os.environ["MINIMAX_API_KEY"] = "MINIMAX_API_KEY"


embeddings = MiniMaxEmbeddings()

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