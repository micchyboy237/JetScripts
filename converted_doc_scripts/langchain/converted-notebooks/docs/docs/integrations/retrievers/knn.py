from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.retrievers import KNNRetriever
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
# kNN

>In statistics, the [k-nearest neighbours algorithm (k-NN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is a non-parametric supervised learning method first developed by `Evelyn Fix` and `Joseph Hodges` in 1951, and later expanded by `Thomas Cover`. It is used for classification and regression.

This notebook goes over how to use a retriever that under the hood uses a kNN.

Largely based on the code of [Andrej Karpathy](https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.html).
"""
logger.info("# kNN")


"""
## Create New Retriever with Texts
"""
logger.info("## Create New Retriever with Texts")

retriever = KNNRetriever.from_texts(
    ["foo", "bar", "world", "hello", "foo bar"], OllamaEmbeddings(
        model="mxbai-embed-large")
)

"""
## Use Retriever

We can now use the retriever!
"""
logger.info("## Use Retriever")

result = retriever.invoke("foo")

result

logger.info("\n\n[DONE]", bright=True)
