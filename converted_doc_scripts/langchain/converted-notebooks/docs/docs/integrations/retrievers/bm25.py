from jet.logger import logger
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from nltk.tokenize import word_tokenize
import nltk
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
# BM25

>[BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25) also known as the `Okapi BM25`, is a ranking function used in information retrieval systems to estimate the relevance of documents to a given search query.
>
>`BM25Retriever` retriever uses the [`rank_bm25`](https://github.com/dorianbrown/rank_bm25) package.
"""
logger.info("# BM25")

# %pip install --upgrade --quiet  rank_bm25


"""
## Create New Retriever with Texts
"""
logger.info("## Create New Retriever with Texts")

retriever = BM25Retriever.from_texts(["foo", "bar", "world", "hello", "foo bar"])

"""
## Create a New Retriever with Documents

You can now create a new retriever with the documents you created.
"""
logger.info("## Create a New Retriever with Documents")


retriever = BM25Retriever.from_documents(
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
## Preprocessing Function
Pass a custom preprocessing function to the retriever to improve search results. Tokenizing text at the word level can enhance retrieval, especially when using vector stores like Chroma, Pinecone, or Faiss for chunked documents.
"""
logger.info("## Preprocessing Function")


nltk.download("punkt_tab")


retriever = BM25Retriever.from_documents(
    [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="world"),
        Document(page_content="hello"),
        Document(page_content="foo bar"),
    ],
    k=2,
    preprocess_func=word_tokenize,
)

result = retriever.invoke("bar")
result

logger.info("\n\n[DONE]", bright=True)