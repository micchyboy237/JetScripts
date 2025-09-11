from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_vdms.vectorstores import VDMS, VDMS_Client
import EmbeddingTabs from "@theme/EmbeddingTabs";
import logging
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
sidebar_label: VDMS
---

# Intel's Visual Data Management System (VDMS)

This notebook covers how to get started with VDMS as a vector store.

>Intel's [Visual Data Management System (VDMS)](https://github.com/IntelLabs/vdms) is a storage solution for efficient access of big-”visual”-data that aims to achieve cloud scale by searching for relevant visual data via visual metadata stored as a graph and enabling machine friendly enhancements to visual data for faster access. VDMS is licensed under MIT. For more information on `VDMS`, visit [this page](https://github.com/IntelLabs/vdms/wiki), and find the LangChain API reference [here](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.vdms.VDMS.html).

VDMS supports:
* K nearest neighbor search
* Euclidean distance (L2) and inner product (IP)
* Libraries for indexing and computing distances: FaissFlat (Default), FaissHNSWFlat, FaissIVFFlat, Flinng, TileDBDense, TileDBSparse
* Embeddings for text, images, and video
* Vector and metadata searches

## Setup

To access VDMS vector stores you'll need to install the `langchain-vdms` integration package and deploy a VDMS server via the publicly available Docker image.
For simplicity, this notebook will deploy a VDMS server on local host using port 55555.
"""
logger.info("# Intel's Visual Data Management System (VDMS)")

# %pip install -qU "langchain-vdms>=0.1.3"
# !docker run --no-healthcheck --rm -d -p 55555:55555 --name vdms_vs_test_nb intellabs/vdms:latest
# !sleep 5

"""
### Credentials

You can use `VDMS` without any credentials.

To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("### Credentials")



"""
## Initialization
Use the VDMS Client to connect to a VDMS vectorstore using FAISS IndexFlat indexing (default) and Euclidean distance (default) as the distance metric for similarity search.


<EmbeddingTabs/>
"""
logger.info("## Initialization")

# ! pip install -qU langchain-huggingface

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


collection_name = "test_collection_faiss_L2"

vdms_client = VDMS_Client(host="localhost", port=55555)

vector_store = VDMS(
    client=vdms_client,
    embedding=embeddings,
    collection_name=collection_name,
    engine="FaissFlat",
    distance_strategy="L2",
)

"""
## Manage vector store

### Add items to vector store
"""
logger.info("## Manage vector store")


logging.basicConfig()
logging.getLogger("langchain_vdms.vectorstores").setLevel(logging.INFO)


document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]

doc_ids = [str(i) for i in range(1, 11)]
vector_store.add_documents(documents=documents, ids=doc_ids)

"""
If an id is provided multiple times, `add_documents` does not check whether the ids are unique. For this reason, use `upsert` to delete existing id entries prior to adding.
"""
logger.info("If an id is provided multiple times, `add_documents` does not check whether the ids are unique. For this reason, use `upsert` to delete existing id entries prior to adding.")

vector_store.upsert(documents, ids=doc_ids)

"""
### Update items in vector store
"""
logger.info("### Update items in vector store")

updated_document_1 = Document(
    page_content="I had chocolate chip pancakes and fried eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

updated_document_2 = Document(
    page_content="The weather forecast for tomorrow is sunny and warm, with a high of 82 degrees.",
    metadata={"source": "news"},
    id=2,
)

vector_store.update_documents(
    ids=doc_ids[:2],
    documents=[updated_document_1, updated_document_2],
    batch_size=2,
)

"""
### Delete items from vector store
"""
logger.info("### Delete items from vector store")

vector_store.delete(ids=doc_ids[-1])

"""
## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.

### Query directly

Performing a simple similarity search can be done as follows:
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": ["==", "tweet"]},
)
for doc in results:
    logger.debug(f"* ID={doc.id}: {doc.page_content} [{doc.metadata}]")

"""
If you want to execute a similarity search and receive the corresponding scores you can run:
"""
logger.info("If you want to execute a similarity search and receive the corresponding scores you can run:")

results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": ["==", "news"]}
)
for doc, score in results:
    logger.debug(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

"""
If you want to execute a similarity search using an embedding you can run:
"""
logger.info("If you want to execute a similarity search using an embedding you can run:")

results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("I love green eggs and ham!"), k=1
)
for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

"""
### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.
"""
logger.info("### Query by turning into retriever")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
results = retriever.invoke("Stealing from the bank is a crime")
for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 1,
        "score_threshold": 0.0,  # >= score_threshold
    },
)
results = retriever.invoke("Stealing from the bank is a crime")
for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "fetch_k": 10},
)
results = retriever.invoke(
    "Stealing from the bank is a crime", filter={"source": ["==", "news"]}
)
for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

"""
### Delete collection
Previously, we removed documents based on its `id`. Here, all documents are removed since no ID is provided.
"""
logger.info("### Delete collection")

logger.debug("Documents before deletion: ", vector_store.count())

vector_store.delete(collection_name=collection_name)

logger.debug("Documents after deletion: ", vector_store.count())

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Multi-modal RAG using VDMS](https://github.com/langchain-ai/langchain/blob/master/cookbook/multi_modal_RAG_vdms.ipynb)
- [Visual RAG using VDMS](https://github.com/langchain-ai/langchain/blob/master/cookbook/visual_RAG_vdms.ipynb)
- [Tutorials](/docs/tutorials/)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/#retrieval)

## Similarity Search using other engines

VDMS supports various libraries for indexing and computing distances: FaissFlat (Default), FaissHNSWFlat, FaissIVFFlat, Flinng, TileDBDense, and TileDBSparse.
By default, the vectorstore uses FaissFlat. Below we show a few examples using the other engines.

### Similarity Search using Faiss HNSWFlat and Euclidean Distance

Here, we add the documents to VDMS using Faiss IndexHNSWFlat indexing and L2 as the distance metric for similarity search. We search for three documents (`k=3`) related to a query and also return the score along with the document.
"""
logger.info("## Usage for retrieval-augmented generation")

db_FaissHNSWFlat = VDMS.from_documents(
    documents,
    client=vdms_client,
    ids=doc_ids,
    collection_name="my_collection_FaissHNSWFlat_L2",
    embedding=embeddings,
    engine="FaissHNSWFlat",
    distance_strategy="L2",
)
k = 3
query = "LangChain provides abstractions to make working with LLMs easy"
docs_with_score = db_FaissHNSWFlat.similarity_search_with_score(query, k=k, filter=None)

for res, score in docs_with_score:
    logger.debug(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

"""
### Similarity Search using Faiss IVFFlat and Inner Product (IP) Distance

We add the documents to VDMS using Faiss IndexIVFFlat indexing and IP as the distance metric for similarity search. We search for three documents (`k=3`) related to a query and also return the score along with the document.
"""
logger.info("### Similarity Search using Faiss IVFFlat and Inner Product (IP) Distance")

db_FaissIVFFlat = VDMS.from_documents(
    documents,
    client=vdms_client,
    ids=doc_ids,
    collection_name="my_collection_FaissIVFFlat_IP",
    embedding=embeddings,
    engine="FaissIVFFlat",
    distance_strategy="IP",
)

k = 3
query = "LangChain provides abstractions to make working with LLMs easy"
docs_with_score = db_FaissIVFFlat.similarity_search_with_score(query, k=k, filter=None)
for res, score in docs_with_score:
    logger.debug(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

"""
### Similarity Search using FLINNG and IP Distance

In this section, we add the documents to VDMS using Filters to Identify Near-Neighbor Groups (FLINNG) indexing and IP as the distance metric for similarity search. We search for three documents (`k=3`) related to a query and also return the score along with the document.
"""
logger.info("### Similarity Search using FLINNG and IP Distance")

db_Flinng = VDMS.from_documents(
    documents,
    client=vdms_client,
    ids=doc_ids,
    collection_name="my_collection_Flinng_IP",
    embedding=embeddings,
    engine="Flinng",
    distance_strategy="IP",
)
k = 3
query = "LangChain provides abstractions to make working with LLMs easy"
docs_with_score = db_Flinng.similarity_search_with_score(query, k=k, filter=None)
for res, score in docs_with_score:
    logger.debug(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

"""
## Filtering on metadata

It can be helpful to narrow down the collection before working with it.

For example, collections can be filtered on metadata using the `get_by_constraints` method. A dictionary is used to filter metadata. Here we retrieve the document where `langchain_id = "2"` and remove it from the vector store.

***NOTE:*** `id` was generated as additional metadata as an integer while `langchain_id` (the internal ID) is an unique string for each entry.
"""
logger.info("## Filtering on metadata")

response, response_array = db_FaissIVFFlat.get_by_constraints(
    db_FaissIVFFlat.collection_name,
    limit=1,
    include=["metadata", "embeddings"],
    constraints={"langchain_id": ["==", "2"]},
)

db_FaissIVFFlat.delete(collection_name=db_FaissIVFFlat.collection_name, ids=["2"])

logger.debug("Deleted entry:")
for doc in response:
    logger.debug(f"* ID={doc.id}: {doc.page_content} [{doc.metadata}]")

response, response_array = db_FaissIVFFlat.get_by_constraints(
    db_FaissIVFFlat.collection_name,
    include=["metadata"],
)
for doc in response:
    logger.debug(f"* ID={doc.id}: {doc.page_content} [{doc.metadata}]")

"""
Here we use `id` to filter for a range of IDs since it is an integer.
"""
logger.info("Here we use `id` to filter for a range of IDs since it is an integer.")

response, response_array = db_FaissIVFFlat.get_by_constraints(
    db_FaissIVFFlat.collection_name,
    include=["metadata", "embeddings"],
    constraints={"source": ["==", "news"]},
)
for doc in response:
    logger.debug(f"* ID={doc.id}: {doc.page_content} [{doc.metadata}]")

"""
## Stop VDMS Server
"""
logger.info("## Stop VDMS Server")

# !docker kill vdms_vs_test_nb

"""
## API reference

TODO: add API reference


"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)