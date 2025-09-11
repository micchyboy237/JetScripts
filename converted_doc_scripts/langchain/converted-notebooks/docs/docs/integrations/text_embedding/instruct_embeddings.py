from jet.logger import logger
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
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
# Instruct Embeddings on Hugging Face

>[Hugging Face sentence-transformers](https://huggingface.co/sentence-transformers) is a Python framework for state-of-the-art sentence, text and image embeddings.
>One of the instruct embedding models is used in the `HuggingFaceInstructEmbeddings` class.
"""
logger.info("# Instruct Embeddings on Hugging Face")


embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)

text = "This is a test document."

query_result = embeddings.embed_query(text)

logger.info("\n\n[DONE]", bright=True)