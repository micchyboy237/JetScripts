from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
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
---
sidebar_position: 0
---

# How to use a vectorstore as a retriever

A vector store retriever is a [retriever](/docs/concepts/retrievers/) that uses a [vector store](/docs/concepts/vectorstores/) to retrieve documents. It is a lightweight wrapper around the vector store class to make it conform to the retriever [interface](/docs/concepts/runnables/).
It uses the search methods implemented by a vector store, like similarity search and MMR, to query the texts in the vector store.

In this guide we will cover:

1. How to instantiate a retriever from a vectorstore;
2. How to specify the search type for the retriever;
3. How to specify additional search parameters, such as threshold scores and top-k.

## Creating a retriever from a vectorstore

You can build a retriever from a vectorstore using its [.as_retriever](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore.as_retriever) method. Let's walk through an example.

First we instantiate a vectorstore. We will use an in-memory [FAISS](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html) vectorstore:
"""
logger.info("# How to use a vectorstore as a retriever")


loader = TextLoader("state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = FAISS.from_documents(texts, embeddings)

"""
We can then instantiate a retriever:
"""
logger.info("We can then instantiate a retriever:")

retriever = vectorstore.as_retriever()

"""
This creates a retriever (specifically a [VectorStoreRetriever](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html)), which we can use in the usual way:
"""
logger.info(
    "This creates a retriever (specifically a [VectorStoreRetriever](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html)), which we can use in the usual way:")

docs = retriever.invoke(
    "what did the president say about ketanji brown jackson?")

"""
## Maximum marginal relevance retrieval
By default, the vector store retriever uses similarity search. If the underlying vector store supports maximum marginal relevance search, you can specify that as the search type.

This effectively specifies what method on the underlying vectorstore is used (e.g., `similarity_search`, `max_marginal_relevance_search`, etc.).
"""
logger.info("## Maximum marginal relevance retrieval")

retriever = vectorstore.as_retriever(search_type="mmr")

docs = retriever.invoke(
    "what did the president say about ketanji brown jackson?")

"""
## Passing search parameters

We can pass parameters to the underlying vectorstore's search methods using `search_kwargs`.

### Similarity score threshold retrieval

For example, we can set a similarity score threshold and only return documents with a score above that threshold.
"""
logger.info("## Passing search parameters")

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)

docs = retriever.invoke(
    "what did the president say about ketanji brown jackson?")

"""
### Specifying top k

We can also limit the number of documents `k` returned by the retriever.
"""
logger.info("### Specifying top k")

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

docs = retriever.invoke(
    "what did the president say about ketanji brown jackson?")
len(docs)

logger.info("\n\n[DONE]", bright=True)
