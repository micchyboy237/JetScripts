from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.retrievers import SVMRetriever
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
# SVM

>[Support vector machines (SVMs)](https://scikit-learn.org/stable/modules/svm.html#support-vector-machines) are a set of supervised learning methods used for classification, regression and outliers detection.

This notebook goes over how to use a retriever that under the hood uses an `SVM` using `scikit-learn` package.

Largely based on https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.html
"""
logger.info("# SVM")

# %pip install --upgrade --quiet  scikit-learn

# %pip install --upgrade --quiet  lark

"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info(
    "We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")


"""
## Create New Retriever with Texts
"""
logger.info("## Create New Retriever with Texts")

retriever = SVMRetriever.from_texts(
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
