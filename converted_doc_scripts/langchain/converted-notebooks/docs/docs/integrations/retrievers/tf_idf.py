from jet.logger import logger
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document
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
# TF-IDF

>[TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) means term-frequency times inverse document-frequency.

This notebook goes over how to use a retriever that under the hood uses [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) using `scikit-learn` package.

For more information on the details of TF-IDF see [this blog post](https://medium.com/data-science-bootcamp/tf-idf-basics-of-information-retrieval-48de122b2a4c).
"""
logger.info("# TF-IDF")

# %pip install --upgrade --quiet  scikit-learn


"""
## Create New Retriever with Texts
"""
logger.info("## Create New Retriever with Texts")

retriever = TFIDFRetriever.from_texts(["foo", "bar", "world", "hello", "foo bar"])

"""
## Create a New Retriever with Documents

You can now create a new retriever with the documents you created.
"""
logger.info("## Create a New Retriever with Documents")


retriever = TFIDFRetriever.from_documents(
    [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="world"),
        Document(page_content="hello"),
        Document(page_content="foo bar"),
    ]
)

"""
## Use Retriever

We can now use the retriever!
"""
logger.info("## Use Retriever")

result = retriever.invoke("foo")

result

"""
## Save and load

You can easily save and load this retriever, making it handy for local development!
"""
logger.info("## Save and load")

retriever.save_local("testing.pkl")

retriever_copy = TFIDFRetriever.load_local("testing.pkl")

retriever_copy.invoke("foo")

logger.info("\n\n[DONE]", bright=True)