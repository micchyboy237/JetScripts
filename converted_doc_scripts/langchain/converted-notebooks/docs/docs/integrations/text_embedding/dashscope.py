from jet.logger import logger
from langchain_community.embeddings import DashScopeEmbeddings
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
# DashScope

Let's load the DashScope Embedding class.
"""
logger.info("# DashScope")


embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_
)

text = "This is a test document."

query_result = embeddings.embed_query(text)
logger.debug(query_result)

doc_results = embeddings.embed_documents(["foo"])
logger.debug(doc_results)

logger.info("\n\n[DONE]", bright=True)