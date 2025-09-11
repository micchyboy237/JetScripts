from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_core.documents import Document
from langchain_postgres import PGVector
import EmbeddingTabs from "@theme/EmbeddingTabs"
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
# PGVector

> An implementation of LangChain vectorstore abstraction using `postgres` as the backend and utilizing the `pgvector` extension.

The code lives in an integration package called: [langchain-postgres](https://github.com/langchain-ai/langchain-postgres/).

## Status

This code has been ported over from `langchain-community` into a dedicated package called `langchain-postgres`. The following changes have been made:

* `langchain-postgres` works only with psycopg3. Please update your connnecion strings from `postgresql+psycopg2://...` to `postgresql+psycopg://langchain:langchain@...` (yes, it's the driver name is `psycopg` not `psycopg3`, but it'll use `psycopg3`.
* The schema of the embedding store and collection have been changed to make add_documents work correctly with user specified ids.
* One has to pass an explicit connection object now.


Currently, there is **no mechanism** that supports easy data migration on schema changes. So any schema changes in the vectorstore will require the user to recreate the tables and re-add the documents.
If this is a concern, please use a different vectorstore. If not, this implementation should be fine for your use case.

## Setup

First donwload the partner package:
"""
logger.info("# PGVector")

pip install - qU langchain-postgres

"""
You can run the following command to spin up a postgres container with the `pgvector` extension:
"""
logger.info(
    "You can run the following command to spin up a postgres container with the `pgvector` extension:")

# %docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16

"""
### Credentials

There are no credentials needed to run this notebook, just make sure you downloaded the `langchain-postgres` package and correctly started the postgres container.

If you want to get best in-class automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("### Credentials")


"""
## Instantiation


<EmbeddingTabs/>
"""
logger.info("## Instantiation")


embeddings = OllamaEmbeddings(model="nomic-embed-text")


# Uses psycopg3!
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

"""
## Manage vector store

### Add items to vector store

Note that adding documents by ID will over-write any existing documents that match that ID.
"""
logger.info("## Manage vector store")


docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
]

vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

"""
### Delete items from vector store
"""
logger.info("### Delete items from vector store")

vector_store.delete(ids=["3"])

"""
## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent. 

### Filtering Support

The vectorstore supports a set of filters that can be applied against the metadata fields of the documents.

| Operator | Meaning/Category        |
|----------|-------------------------|
| \$eq      | Equality (==)           |
| \$ne      | Inequality (!=)         |
| \$lt      | Less than (&lt;)           |
| \$lte     | Less than or equal (&lt;=) |
| \$gt      | Greater than (>)        |
| \$gte     | Greater than or equal (>=) |
| \$in      | Special Cased (in)      |
| \$nin     | Special Cased (not in)  |
| \$between | Special Cased (between) |
| \$like    | Text (like)             |
| \$ilike   | Text (case-insensitive like) |
| \$and     | Logical (and)           |
| \$or      | Logical (or)            |

### Query directly

Performing a simple similarity search can be done as follows:
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(
    "kitty", k=10, filter={"id": {"$in": [1, 5, 2, 9]}}
)
for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

"""
If you provide a dict with multiple fields, but no operators, the top level will be interpreted as a logical **AND** filter
"""
logger.info("If you provide a dict with multiple fields, but no operators, the top level will be interpreted as a logical **AND** filter")

vector_store.similarity_search(
    "ducks",
    k=10,
    filter={"id": {"$in": [1, 5, 2, 9]},
            "location": {"$in": ["pond", "market"]}},
)

vector_store.similarity_search(
    "ducks",
    k=10,
    filter={
        "$and": [
            {"id": {"$in": [1, 5, 2, 9]}},
            {"location": {"$in": ["pond", "market"]}},
        ]
    },
)

"""
If you want to execute a similarity search and receive the corresponding scores you can run:
"""
logger.info(
    "If you want to execute a similarity search and receive the corresponding scores you can run:")

results = vector_store.similarity_search_with_score(query="cats", k=1)
for doc, score in results:
    logger.debug(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

"""
For a full list of the different searches you can execute on a `PGVector` vector store, please refer to the [API reference](https://python.langchain.com/api_reference/postgres/vectorstores/langchain_postgres.vectorstores.PGVector.html).

### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.
"""
logger.info("### Query by turning into retriever")

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1})
retriever.invoke("kitty")

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/rag)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

## API reference

For detailed documentation of all __ModuleName__VectorStore features and configurations head to the API reference: https://python.langchain.com/api_reference/postgres/vectorstores/langchain_postgres.vectorstores.PGVector.html
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)
