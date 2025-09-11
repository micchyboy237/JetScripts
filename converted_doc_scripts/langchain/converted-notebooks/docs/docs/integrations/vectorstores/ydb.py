from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_core.documents import Document
from langchain_ydb.vectorstores import YDB, YDBSearchStrategy, YDBSettings
from uuid import uuid4
import EmbeddingTabs from "@theme/EmbeddingTabs";
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
# YDB

> [YDB](https://ydb.tech/) is a versatile open source Distributed SQL Database that combines high availability and scalability with strong consistency and ACID transactions. It accommodates transactional (OLTP), analytical (OLAP), and streaming workloads simultaneously.

This notebook shows how to use functionality related to the `YDB` vector store.

## Setup

First, set up a local YDB with Docker:
"""
logger.info("# YDB")

# ! docker run -d -p 2136:2136 --name ydb-langchain -e YDB_USE_IN_MEMORY_PDISKS=true -h localhost ydbplatform/local-ydb:trunk

"""
You'll need to install `langchain-ydb` to use this integration
"""
logger.info("You'll need to install `langchain-ydb` to use this integration")

# ! pip install -qU langchain-ydb

"""
### Credentials

There are no credentials for this notebook, just make sure you have installed the packages as shown above.

If you want to get best in-class automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("### Credentials")



"""
## Initialization


<EmbeddingTabs/>
"""
logger.info("## Initialization")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")


settings = YDBSettings(
    table="ydb_example",
    strategy=YDBSearchStrategy.COSINE_SIMILARITY,
)
vector_store = YDB(embeddings, config=settings)

"""
## Manage vector store

Once you have created your vector store, you can interact with it by adding and deleting different items.

### Add items to vector store

Prepare documents to work with:
"""
logger.info("## Manage vector store")



document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
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
uuids = [str(uuid4()) for _ in range(len(documents))]

"""
You can add items to your vector store by using the `add_documents` function.
"""
logger.info("You can add items to your vector store by using the `add_documents` function.")

vector_store.add_documents(documents=documents, ids=uuids)

"""
### Delete items from vector store

You can delete items from your vector store by ID using the `delete` function.
"""
logger.info("### Delete items from vector store")

vector_store.delete(ids=[uuids[-1]])

"""
## Query vector store

Once your vector store has been created and relevant documents have been added, you will likely want to query it during the execution of your chain or agent.

### Query directly

#### Similarity search

A simple similarity search can be performed as follows:
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy", k=2
)
for res in results:
    logger.debug(f"* {res.page_content} [{res.metadata}]")

"""
#### Similarity search with score

You can also perform a search with a score:
"""
logger.info("#### Similarity search with score")

results = vector_store.similarity_search_with_score("Will it be hot tomorrow?", k=3)
for res, score in results:
    logger.debug(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")

"""
### Filtering

You can search with filters as described below:
"""
logger.info("### Filtering")

results = vector_store.similarity_search_with_score(
    "What did I eat for breakfast?",
    k=4,
    filter={"source": "tweet"},
)
for res, _ in results:
    logger.debug(f"* {res.page_content} [{res.metadata}]")

"""
### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.

Here's how to transform your vector store into a retriever and then invoke the retriever with a simple query and filter.
"""
logger.info("### Query by turning into retriever")

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2},
)
results = retriever.invoke(
    "Stealing from the bank is a crime", filter={"source": "news"}
)
for res in results:
    logger.debug(f"* {res.page_content} [{res.metadata}]")

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

## API reference

For detailed documentation of all `YDB` features and configurations head to the API reference:https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.ydb.YDB.html
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)