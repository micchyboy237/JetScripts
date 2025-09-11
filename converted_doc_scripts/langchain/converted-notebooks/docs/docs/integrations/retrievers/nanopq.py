from jet.logger import logger
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.retrievers import NanoPQRetriever
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
# NanoPQ (Product Quantization)

>[Product Quantization algorithm (k-NN)](https://towardsdatascience.com/similarity-search-product-quantization-b2a1a6397701) in brief is a quantization algorithm that helps in compression of database vectors which helps in semantic search when large datasets are involved. In a nutshell, the embedding is split into M subspaces which further goes through clustering. Upon clustering the vectors the centroid vector gets mapped to the vectors present in the each of the clusters of the subspace. 

This notebook goes over how to use a retriever that under the hood uses a Product Quantization which has been implemented by the [nanopq](https://github.com/matsui528/nanopq) package.
"""
logger.info("# NanoPQ (Product Quantization)")

# %pip install -qU langchain-community langchain-ollama nanopq


"""
## Create New Retriever with Texts
"""
logger.info("## Create New Retriever with Texts")

retriever = NanoPQRetriever.from_texts(
    ["Great world", "great words", "world", "planets of the world"],
    SpacyEmbeddings(model_name="en_core_web_sm"),
    clusters=2,
    subspace=2,
)

"""
## Use Retriever

We can now use the retriever!
"""
logger.info("## Use Retriever")

retriever.invoke("earth")

logger.info("\n\n[DONE]", bright=True)