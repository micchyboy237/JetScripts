from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_core.documents import Document
from langchain_surrealdb.vectorstores import SurrealDBVectorStore
from surrealdb import Surreal
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
# SurrealDBVectorStore

> [SurrealDB](https://surrealdb.com) is a unified, multi-model database purpose-built for AI systems. It combines structured and unstructured data (including vector search, graph traversal, relational queries, full-text search, document storage, and time-series data) into a single ACID-compliant engine, scaling from a 3 MB edge binary to petabyte-scale clusters in the cloud. By eliminating the need for multiple specialized stores, SurrealDB simplifies architectures, reduces latency, and ensures consistency for AI workloads.
>
> **Why SurrealDB Matters for GenAI Systems**
> - **One engine for storage and memory:** Combine durable storage and fast, agent-friendly memory in a single system, providing all the data your agent needs and removing the need to sync multiple systems.
> - **One-hop memory for agents:** Run vector search, graph traversal, semantic joins, and transactional writes in a single query, giving LLM agents fast, consistent memory access without stitching relational, graph and vector databases together.
> - **In-place inference and real-time updates:** SurrealDB enables agents to run inference next to data and receive millisecond-fresh updates, critical for real-time reasoning and collaboration.
> - **Versioned, durable context:** SurrealDB supports time-travel queries and versioned records, letting agents audit or “replay” past states for consistent, explainable reasoning.
> - **Plug-and-play agent memory:** Expose AI memory as a native concept, making it easy to use SurrealDB as a drop-in backend for AI frameworks.

This notebook covers how to get started with the SurrealDB vector store.

## Setup

You can run SurrealDB locally or start with a [free SurrealDB cloud account](https://surrealdb.com/docs/cloud/getting-started).

For local, two options:
1. [Install SurrealDB](https://surrealdb.com/docs/surrealdb/installation) and [run SurrealDB](https://surrealdb.com/docs/surrealdb/installation/running). Run in-memory with:

    ```bash
    surreal start -u root -p root
    ```

2. [Run with Docker](https://surrealdb.com/docs/surrealdb/installation/running/docker).

    ```bash
    docker run --rm --pull always -p 8000:8000 surrealdb/surrealdb:latest start
    ```

## Install dependencies

Install `langchain-surrealdb` and `surrealdb` python packages.

```shell
# -- Using pip
pip install --upgrade langchain-surrealdb surrealdb
# -- Using poetry
poetry add langchain-surrealdb surrealdb
# -- Using uv
uv add --upgrade langchain-surrealdb surrealdb
```

To run this notebook, we just need to install the additional dependencies required by this example:
"""
logger.info("# SurrealDBVectorStore")

# !poetry add --quiet --group docs langchain-ollama langchain-surrealdb

"""
#
#
 
I
n
i
t
i
a
l
i
z
a
t
i
o
n
"""
logger.info("#")


conn = Surreal("ws://localhost:8000/rpc")
conn.signin({"username": "root", "password": "root"})
conn.use("langchain", "demo")
vector_store = SurrealDBVectorStore(OllamaEmbeddings(model="mxbai-embed-large"), conn)

"""
## Manage vector store

### Add items to vector store
"""
logger.info("## Manage vector store")


_url = "https://surrealdb.com"
d1 = Document(page_content="foo", metadata={"source": _url})
d2 = Document(page_content="SurrealDB", metadata={"source": _url})
d3 = Document(page_content="bar", metadata={"source": _url})
d4 = Document(page_content="this is surreal", metadata={"source": _url})

vector_store.add_documents(documents=[d1, d2, d3, d4], ids=["1", "2", "3", "4"])

"""
#
#
#
 
U
p
d
a
t
e
 
i
t
e
m
s
 
i
n
 
v
e
c
t
o
r
 
s
t
o
r
e
"""
logger.info("#")

updated_document = Document(
    page_content="zar", metadata={"source": "https://example.com"}
)

vector_store.add_documents(documents=[updated_document], ids=["3"])

"""
#
#
#
 
D
e
l
e
t
e
 
i
t
e
m
s
 
f
r
o
m
 
v
e
c
t
o
r
 
s
t
o
r
e
"""
logger.info("#")

vector_store.delete(ids=["3"])

"""
## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent. 

### Query directly

Performing a simple similarity search can be done as follows:
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(
    query="surreal", k=1, custom_filter={"source": "https://surrealdb.com"}
)
for doc in results:
    logger.debug(f"{doc.page_content} [{doc.metadata}]")  # noqa: T201

"""
I
f
 
y
o
u
 
w
a
n
t
 
t
o
 
e
x
e
c
u
t
e
 
a
 
s
i
m
i
l
a
r
i
t
y
 
s
e
a
r
c
h
 
a
n
d
 
r
e
c
e
i
v
e
 
t
h
e
 
c
o
r
r
e
s
p
o
n
d
i
n
g
 
s
c
o
r
e
s
 
y
o
u
 
c
a
n
 
r
u
n
:
"""
logger.info("I")

results = vector_store.similarity_search_with_score(
    query="thud", k=1, custom_filter={"source": "https://surrealdb.com"}
)
for doc, score in results:
    logger.debug(f"[similarity={score:.0%}] {doc.page_content}")  # noqa: T201

"""
### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.
"""
logger.info("### Query by turning into retriever")

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "lambda_mult": 0.5}
)
retriever.invoke("surreal")

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval/)

## API reference

For detailed documentation of all SurrealDBVectorStore features and configurations head to the API reference: https://python.langchain.com/api_reference/surrealdb/index.html

## Next steps

- look at the [basic example](https://github.com/surrealdb/langchain-surrealdb/tree/main/examples/basic). Use the Dockerfile to try it out!
- look at the [graph example](https://github.com/surrealdb/langchain-surrealdb/tree/main/examples/graph)
- try the [jupyther notebook](https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/surrealdb.ipynb)
- [Awesome SurrealDB](https://github.com/surrealdb/awesome-surreal), A curated list of SurrealDB resources, tools, utilities, and applications
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)