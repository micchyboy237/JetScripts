from jet.logger import logger
from langchain_community.embeddings import AwaEmbeddings
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
# AwaDB

>[AwaDB](https://github.com/awa-ai/awadb) is an AI Native database for the search and storage of embedding vectors used by LLM Applications.

This notebook explains how to use `AwaEmbeddings` in LangChain.
"""
logger.info("# AwaDB")



"""
## import the library
"""
logger.info("## import the library")


Embedding = AwaEmbeddings()

"""
# Set embedding model
Users can use `Embedding.set_model()` to specify the embedding model. \
The input of this function is a string which represents the model's name. \
The list of currently supported models can be obtained [here](https://github.com/awa-ai/awadb) \ \ 

The **default model** is `all-mpnet-base-v2`, it can be used without setting.
"""
logger.info("# Set embedding model")

text = "our embedding test"

Embedding.set_model("all-mpnet-base-v2")

res_query = Embedding.embed_query("The test information")
res_document = Embedding.embed_documents(["test1", "another test"])

logger.info("\n\n[DONE]", bright=True)