from jet.logger import logger
from langchain_community.retrievers import SVMRetriever
from langchain_community.vectorstores import SKLearnVectorStore
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
# scikit-learn

>[scikit-learn](https://scikit-learn.org/stable/) is an open-source collection of machine learning algorithms,
> including some implementations of the [k nearest neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). `SKLearnVectorStore` wraps this implementation and adds the possibility to persist the vector store in json, bson (binary json) or Apache Parquet format.

## Installation and Setup

- Install the Python package with `pip install scikit-learn`


## Vector Store

`SKLearnVectorStore` provides a simple wrapper around the nearest neighbor implementation in the
scikit-learn package, allowing you to use it as a vectorstore.

To import this vectorstore:
"""
logger.info("# scikit-learn")


"""
For a more detailed walkthrough of the SKLearnVectorStore wrapper, see [this notebook](/docs/integrations/vectorstores/sklearn).


## Retriever

`Support vector machines (SVMs)` are the supervised learning
methods used for classification, regression and outliers detection.

See a [usage example](/docs/integrations/retrievers/svm).
"""
logger.info("## Retriever")


logger.info("\n\n[DONE]", bright=True)