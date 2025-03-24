from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document

initialize_ollama_settings()

"""
# TF-IDF

>[TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) means term-frequency times inverse document-frequency.

This notebook goes over how to use a retriever that under the hood uses [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) using `scikit-learn` package.

For more information on the details of TF-IDF see [this blog post](https://medium.com/data-science-bootcamp/tf-idf-basics-of-information-retrieval-48de122b2a4c).
"""

# %pip install --upgrade --quiet  scikit-learn


"""
## Create New Retriever with Texts
"""

retriever = TFIDFRetriever.from_texts(
    ["foo", "bar", "world", "hello", "foo bar"])

"""
## Create a New Retriever with Documents

You can now create a new retriever with the documents you created.
"""


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

result = retriever.invoke("foo")

logger.newline()
logger.info("Result 1")
logger.success(result)

"""
## Save and load

You can easily save and load this retriever, making it handy for local development!
"""

retriever.save_local("testing.pkl")

retriever_copy = TFIDFRetriever.load_local(
    "testing.pkl", allow_dangerous_deserialization=True)

logger.newline()
logger.info("Result 2")
logger.success(retriever_copy.invoke("foo"))

logger.info("\n\n[DONE]", bright=True)
