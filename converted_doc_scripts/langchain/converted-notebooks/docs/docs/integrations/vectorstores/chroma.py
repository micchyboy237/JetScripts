from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
import EmbeddingTabs from "@theme/EmbeddingTabs";
import chromadb
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
# Chroma

This notebook covers how to get started with the `Chroma` vector store.

>[Chroma](https://docs.trychroma.com/getting-started) is a AI-native open-source vector database focused on developer productivity and happiness. Chroma is licensed under Apache 2.0. View the full docs of `Chroma` at [this page](https://docs.trychroma.com/integrations/frameworks/langchain), and find the API reference for the LangChain integration at [this page](https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html).

:::info Chroma Cloud

Chroma Cloud powers serverless vector and full-text search. It's extremely fast, cost-effective, scalable and painless. Create a DB and try it out in under 30 seconds with $5 of free credits.

[Get started with Chroma Cloud](https://trychroma.com/signup)
:::

## Setup

To access `Chroma` vector stores you'll need to install the `langchain-chroma` integration package.
"""
logger.info("# Chroma")

pip install -qU "langchain-chroma>=0.1.2"

"""
### Credentials

You can use the `Chroma` vector store without any credentials, simply installing the package above is enough!

If you are a [Chroma Cloud](https://trychroma.com/signup) user, set your `CHROMA_TENANT`, `CHROMA_DATABASE`, and `CHROMA_API_KEY` environment variables.

When you install the `chromadb` package you also get access to the Chroma CLI, which can set these for you. First, [login](https://docs.trychroma.com/docs/cli/login) via the CLI, and then use the [`connect` command](https://docs.trychroma.com/docs/cli/db):

```bash
chroma db connect [db_name] --env-file
```

If you want to get best in-class automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("### Credentials")



"""
## Initialization

### Basic Initialization 

Below is a basic initialization, including the use of a directory to save the data locally.


<EmbeddingTabs/>
"""
logger.info("## Initialization")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
#### Running Locally (In-Memory)

You can get a Chroma server running in memory by simply instantiating a `Chroma` instance with a collection name and your embeddings provider:
"""
logger.info("#### Running Locally (In-Memory)")


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)

"""
If you don't need data persistence, this is a great option for experimenting while building your AI application with Langchain.

#### Running Locally (with Data Persistence)

You can provide the `persist_directory` argument to save your data across multiple runs of your program:
"""
logger.info("#### Running Locally (with Data Persistence)")


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

"""
#### Connecting to a Chroma Server

If you have a Chroma server running locally, or you have [deployed](https://docs.trychroma.com/guides/deploy/client-server-mode) one yourself, you can connect to it by providing the `host` argument.

For example, you can start a Chroma server running locally with `chroma run`, and then connect it with `host='localhost'`:
"""
logger.info("#### Connecting to a Chroma Server")


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    host="localhost",
)

"""
For other deployments you can use the `port`, `ssl`, and `headers` arguments to customize your connection.

#### Chroma Cloud

Chroma Cloud users can also build with Langchain. Provide your `Chroma` instance with your Chroma Cloud API key, tenant, and DB name:
"""
logger.info("#### Chroma Cloud")


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

"""
### Initialization from client

You can also initialize from a `Chroma` client, which is particularly useful if you want easier access to the underlying database.

#### Running Locally (In-Memory)
"""
logger.info("### Initialization from client")


client = chromadb.Client()

"""
#### Running Locally (with Data Persistence)
"""
logger.info("#### Running Locally (with Data Persistence)")


client = chromadb.PersistentClient(path="./chroma_langchain_db")

"""
#### Connecting to a Chroma Server

For example, if you are running a Chroma server locally (using `chroma run`):
"""
logger.info("#### Connecting to a Chroma Server")


client = chromadb.HttpClient(host="localhost", port=8000, ssl=False)

"""
#### Chroma Cloud

After setting your `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE`, you can simply instantiate:
"""
logger.info("#### Chroma Cloud")


client = chromadb.CloudClient()

"""
#### Access your Chroma DB
"""
logger.info("#### Access your Chroma DB")

collection = client.get_or_create_collection("collection_name")
collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

"""
#### Create a Chroma Vectorstore
"""
logger.info("#### Create a Chroma Vectorstore")

vector_store_from_client = Chroma(
    client=client,
    collection_name="collection_name",
    embedding_function=embeddings,
)

"""
## Manage vector store

Once you have created your vector store, we can interact with it by adding and deleting different items.

### Add items to vector store

We can add items to our vector store by using the `add_documents` function.
"""
logger.info("## Manage vector store")



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
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)

"""
### Update items in vector store

Now that we have added documents to our vector store, we can update existing documents by using the `update_documents` function.
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

vector_store.update_document(document_id=uuids[0], document=updated_document_1)
vector_store.update_documents(
    ids=uuids[:2], documents=[updated_document_1, updated_document_2]
)

"""
### Delete items from vector store

We can also delete items from our vector store as follows:
"""
logger.info("### Delete items from vector store")

vector_store.delete(ids=uuids[-1])

"""
### Fork a vector store

Forking lets you create a new `Chroma` vector store from an existing one instantly, using copy-on-write under the hood. This means that your new `Chroma` store is identical to the origin, but any modifications to it will not affect the origin, and vice-versa.

Forks are great for any use case that benefits from data versioning. You can learn more about forking in the [Chroma docs](https://docs.trychroma.com/cloud/collection-forking).

Note: Forking is only avaiable on `Chroma` instances with a Chroma Cloud connection.
"""
logger.info("### Fork a vector store")

forked_store = vector_store.fork(new_name="my_forked_collection")

updated_document_2 = Document(
    page_content="The weather forecast for tomorrow is extrmeley hot, with a high of 100 degrees.",
    metadata={"source": "news"},
    id=2,
)

forked_store.update(ids=["2"], documents=[updated_document_2])

"""
## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent. 

### Query directly

#### Similarity search

Performing a simple similarity search can be done as follows:
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    logger.debug(f"* {res.page_content} [{res.metadata}]")

"""
#### Similarity search with score

If you want to execute a similarity search and receive the corresponding scores you can run:
"""
logger.info("#### Similarity search with score")

results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    logger.debug(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

"""
#### Search by vector

You can also search by vector:
"""
logger.info("#### Search by vector")

results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("I love green eggs and ham!"), k=1
)
for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

"""
#### Other search methods

There are a variety of other search methods that are not covered in this notebook. For a full list of the search abilities available for `Chroma` check out the [API reference](https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html).

### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains. For more information on the different search types and kwargs you can pass, please visit the API reference [here](https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.as_retriever).
"""
logger.info("#### Other search methods")

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/rag)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

## API reference

For detailed documentation of all `Chroma` vector store features and configurations head to the API reference: https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)