from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_superlinked import SuperlinkedRetriever
import Link from '@docusaurus/Link';
import os
import shutil
import superlinked.framework as sl


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
title: Superlinked
description: LangChain integration package for the Superlinked retrieval stack
---


### Overview

Superlinked enables context‑aware retrieval using multiple space types (text similarity, categorical, numerical, recency, and more). The `langchain-superlinked` package provides a LangChain‑native `SuperlinkedRetriever` that plugs directly into your RAG chains.

### Links

- <Link to="https://github.com/superlinked/langchain-superlinked">Integration repository</Link>
- <Link to="https://links.superlinked.com/langchain_repo_sl">Superlinked core repository</Link>
- <Link to="https://links.superlinked.com/langchain_article">Article: Build RAG using LangChain & Superlinked</Link>

### Install
"""
logger.info("### Overview")

pip install -U langchain-superlinked superlinked

"""
### Quickstart
"""
logger.info("### Quickstart")


class DocumentSchema(sl.Schema):
    id: sl.IdField
    content: sl.String

doc_schema = DocumentSchema()

text_space = sl.TextSimilaritySpace(
    text=doc_schema.content, model="sentence-transformers/all-MiniLM-L6-v2"
)
doc_index = sl.Index([text_space])

query = (
    sl.Query(doc_index)
    .find(doc_schema)
    .similar(text_space.text, sl.Param("query_text"))
    .select([doc_schema.content])
    .limit(sl.Param("limit"))
)

source = sl.InMemorySource(schema=doc_schema)
executor = sl.InMemoryExecutor(sources=[source], indices=[doc_index])
app = executor.run()
source.put([
    {"id": "1", "content": "Machine learning algorithms process data efficiently."},
    {"id": "2", "content": "Natural language processing understands human language."},
])

retriever = SuperlinkedRetriever(
    sl_client=app, sl_query=query, page_content_field="content"
)

docs = retriever.invoke("artificial intelligence", limit=2)
for d in docs:
    logger.debug(d.page_content)

"""
### What the retriever expects (App and Query)

The retriever takes two core inputs:

- `sl_client`: a Superlinked App created by running an executor (e.g., `InMemoryExecutor(...).run()`)
- `sl_query`: a `QueryDescriptor` returned by chaining `sl.Query(...).find(...).similar(...).select(...).limit(...)`

Minimal setup:
"""
logger.info("### What the retriever expects (App and Query)")


class Doc(sl.Schema):
    id: sl.IdField
    content: sl.String

doc = Doc()
space = sl.TextSimilaritySpace(text=doc.content, model="sentence-transformers/all-MiniLM-L6-v2")
index = sl.Index([space])

query = (
    sl.Query(index)
    .find(doc)
    .similar(space.text, sl.Param("query_text"))
    .select([doc.content])
    .limit(sl.Param("limit"))
)

source = sl.InMemorySource(schema=doc)
app = sl.InMemoryExecutor(sources=[source], indices=[index]).run()

retriever = SuperlinkedRetriever(sl_client=app, sl_query=query, page_content_field="content")

"""
Note: For a persistent vector DB, pass `vector_database=...` to the executor (e.g., Qdrant) before `.run()`.

### Use within a chain
"""
logger.info("### Use within a chain")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    """
    Answer based on context:\n\nContext: {context}\nQuestion: {question}
    """
)

chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
         | prompt
         | ChatOllama(model="llama3.2"))

answer = chain.invoke("How does machine learning work?")

"""
### Resources

- <Link to="https://pypi.org/project/langchain-superlinked/">PyPI: langchain-superlinked</Link>
- <Link to="https://pypi.org/project/superlinked/">PyPI: superlinked</Link>
- <Link to="https://github.com/superlinked/langchain-superlinked">Source repository</Link>
- <Link to="https://links.superlinked.com/langchain_repo_sl">Superlinked core repository</Link>
- <Link to="https://links.superlinked.com/langchain_article">Build RAG using LangChain & Superlinked (article)</Link>
"""
logger.info("### Resources")

logger.info("\n\n[DONE]", bright=True)