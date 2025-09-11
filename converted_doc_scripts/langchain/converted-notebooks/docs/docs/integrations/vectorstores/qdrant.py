from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant import RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
from qdrant_client import models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from qdrant_client.http.models import Distance, VectorParams
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
# Qdrant

>[Qdrant](https://qdrant.tech/documentation/) (read: quadrant) is a vector similarity search engine. It provides a production-ready service with a convenient API to store, search, and manage vectors with additional payload and extended filtering support. It makes it useful for all sorts of neural network or semantic-based matching, faceted search, and other applications.

This documentation demonstrates how to use Qdrant with LangChain for dense (i.e., embedding-based), sparse (i.e., text search) and hybrid retrieval. The `QdrantVectorStore` class supports multiple retrieval modes via Qdrant's new [Query API](https://qdrant.tech/blog/qdrant-1.10.x/). It requires you to run Qdrant v1.10.0 or above.


## Setup

There are various modes of how to run `Qdrant`, and depending on the chosen one, there will be some subtle differences. The options include:
- Local mode, no server required
- Docker deployments
- Qdrant Cloud

Please see the installation instructions [here](https://qdrant.tech/documentation/install/).
"""
logger.info("# Qdrant")

pip install -qU langchain-qdrant

"""
### Credentials

There are no credentials needed to run the code in this notebook.

If you want to get best in-class automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("### Credentials")



"""
## Initialization

### Local mode

The Python client provides the option to run the code in local mode without running the Qdrant server. This is great for testing things out and debugging or storing just a small amount of vectors. The embeddings can be kept fully in-memory or persisted on-disk.

#### In-memory

For some testing scenarios and quick experiments, you may prefer to keep all the data in-memory only, so it gets removed when the client is destroyed - usually at the end of your script/notebook.



<EmbeddingTabs/>
"""
logger.info("## Initialization")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")


client = QdrantClient(":memory:")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings,
)

"""
#### On-disk storage

Local mode, without using the Qdrant server, may also store your vectors on-disk so they persist between runs.
"""
logger.info("#### On-disk storage")

client = QdrantClient(path="/tmp/langchain_qdrant")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings,
)

"""
### On-premise server deployment

No matter if you choose to launch Qdrant locally with [a Docker container](https://qdrant.tech/documentation/install/) or select a Kubernetes deployment with [the official Helm chart](https://github.com/qdrant/qdrant-helm), the way you're going to connect to such an instance will be identical. You'll need to provide a URL pointing to the service.
"""
logger.info("### On-premise server deployment")

url = "<---qdrant url here --->"
docs = []  # put docs here
qdrant = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    collection_name="my_documents",
)

"""
### Qdrant Cloud

If you prefer not to keep yourself busy with managing the infrastructure, you can choose to set up a fully-managed Qdrant cluster on [Qdrant Cloud](https://cloud.qdrant.io/). There is a free forever 1GB cluster included for trying out. The main difference with using a managed version of Qdrant is that you'll need to provide an API key to secure your deployment from being accessed publicly. The value can also be set in a `QDRANT_API_KEY` environment variable.
"""
logger.info("### Qdrant Cloud")

url = "<---qdrant cloud cluster url here --->"

qdrant = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="my_documents",
)

"""
## Using an existing collection

To get an instance of `langchain_qdrant.Qdrant` without loading any new documents or texts, you can use the `Qdrant.from_existing_collection()` method.
"""
logger.info("## Using an existing collection")

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="my_documents",
    url="http://localhost:6333",
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
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees Fahrenheit.",
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

vector_store.add_documents(documents=documents, ids=uuids)

"""
### Delete items from vector store
"""
logger.info("### Delete items from vector store")

vector_store.delete(ids=[uuids[-1]])

"""
## Query vector store

Once your vector store has been created and the relevant documents have been added, you will most likely wish to query it during the running of your chain or agent. 

### Query directly

The simplest scenario for using the Qdrant vector store is to perform a similarity search. Under the hood, our query will be encoded into vector embeddings and used to find similar documents in a Qdrant collection.
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy", k=2
)
for res in results:
    logger.debug(f"* {res.page_content} [{res.metadata}]")

"""
`QdrantVectorStore` supports 3 modes for similarity searches. They can be configured using the `retrieval_mode` parameter.

- Dense Vector Search (default)
- Sparse Vector Search
- Hybrid Search

### Dense Vector Search

Dense vector search involves calculating similarity via vector-based embeddings. To search with only dense vectors:

- The `retrieval_mode` parameter should be set to `RetrievalMode.DENSE`. This is the default behavior.
- A [dense embeddings](https://python.langchain.com/docs/integrations/text_embedding/) value should be provided to the `embedding` parameter.
"""
logger.info("### Dense Vector Search")


client = QdrantClient(path="/tmp/langchain_qdrant")

client.create_collection(
    collection_name="my_documents",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

qdrant = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
)

qdrant.add_documents(documents=documents, ids=uuids)

query = "How much money did the robbers steal?"
found_docs = qdrant.similarity_search(query)
found_docs

"""
### Sparse Vector Search

To search with only sparse vectors:

- The `retrieval_mode` parameter should be set to `RetrievalMode.SPARSE`.
- An implementation of the [`SparseEmbeddings`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/qdrant/langchain_qdrant/sparse_embeddings.py) interface using any sparse embeddings provider has to be provided as a value to the `sparse_embedding` parameter.

The `langchain-qdrant` package provides a [FastEmbed](https://github.com/qdrant/fastembed) based implementation out of the box.

To use it, install the FastEmbed package.
"""
logger.info("### Sparse Vector Search")

# %pip install -qU fastembed


sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

client = QdrantClient(path="/tmp/langchain_qdrant")

client.create_collection(
    collection_name="my_documents",
    vectors_config={"dense": VectorParams(size=3072, distance=Distance.COSINE)},
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
    },
)

qdrant = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.SPARSE,
    sparse_vector_name="sparse",
)

qdrant.add_documents(documents=documents, ids=uuids)

query = "How much money did the robbers steal?"
found_docs = qdrant.similarity_search(query)
found_docs

"""
### Hybrid Vector Search

To perform a hybrid search using dense and sparse vectors with score fusion,

- The `retrieval_mode` parameter should be set to `RetrievalMode.HYBRID`.
- A [dense embeddings](https://python.langchain.com/docs/integrations/text_embedding/) value should be provided to the `embedding` parameter.
- An implementation of the [`SparseEmbeddings`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/qdrant/langchain_qdrant/sparse_embeddings.py) interface using any sparse embeddings provider has to be provided as a value to the `sparse_embedding` parameter.

Note that if you've added documents with the `HYBRID` mode, you can switch to any retrieval mode when searching, since both the dense and sparse vectors are available in the collection.
"""
logger.info("### Hybrid Vector Search")


sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

client = QdrantClient(path="/tmp/langchain_qdrant")

client.create_collection(
    collection_name="my_documents",
    vectors_config={"dense": VectorParams(size=3072, distance=Distance.COSINE)},
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
    },
)

qdrant = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

qdrant.add_documents(documents=documents, ids=uuids)

query = "How much money did the robbers steal?"
found_docs = qdrant.similarity_search(query)
found_docs

"""
If you want to execute a similarity search and receive the corresponding scores you can run:
"""
logger.info("If you want to execute a similarity search and receive the corresponding scores you can run:")

results = vector_store.similarity_search_with_score(
    query="Will it be hot tomorrow", k=1
)
for doc, score in results:
    logger.debug(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

"""
For a full list of all the search functions available for a `QdrantVectorStore`, read the [API reference](https://python.langchain.com/api_reference/qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html)

### Metadata filtering

Qdrant has an [extensive filtering system](https://qdrant.tech/documentation/concepts/filtering/) with rich type support. It is also possible to use the filters in Langchain, by passing an additional param to both the `similarity_search_with_score` and `similarity_search` methods.
"""
logger.info("### Metadata filtering")


results = vector_store.similarity_search(
    query="Who are the best soccer players in the world?",
    k=1,
    filter=models.Filter(
        should=[
            models.FieldCondition(
                key="page_content",
                match=models.MatchValue(
                    value="The top 10 soccer players in the world right now."
                ),
            ),
        ]
    ),
)
for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

"""
### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.
"""
logger.info("### Query by turning into retriever")

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
retriever.invoke("Stealing from the bank is a crime")

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/rag)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

## Customizing Qdrant

There are options to use an existing Qdrant collection within your LangChain application. In such cases, you may need to define how to map Qdrant point into the LangChain `Document`.

### Named vectors

Qdrant supports [multiple vectors per point](https://qdrant.tech/documentation/concepts/collections/#collection-with-multiple-vectors) by named vectors. If you work with a collection created externally or want to have the differently named vector used, you can configure it by providing its name.
"""
logger.info("## Usage for retrieval-augmented generation")


QdrantVectorStore.from_documents(
    docs,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    location=":memory:",
    collection_name="my_documents_2",
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="custom_vector",
    sparse_vector_name="custom_sparse_vector",
)

"""
### Metadata

Qdrant stores your vector embeddings along with the optional JSON-like payload. Payloads are optional, but since LangChain assumes the embeddings are generated from the documents, we keep the context data, so you can extract the original texts as well.

By default, your document is going to be stored in the following payload structure:

```json
{
    "page_content": "Lorem ipsum dolor sit amet",
    "metadata": {
        "foo": "bar"
    }
}
```

You can, however, decide to use different keys for the page content and metadata. That's useful if you already have a collection that you'd like to reuse.
"""
logger.info("### Metadata")

QdrantVectorStore.from_documents(
    docs,
    embeddings,
    location=":memory:",
    collection_name="my_documents_2",
    content_payload_key="my_page_content_key",
    metadata_payload_key="my_meta",
)

"""
## API reference

For detailed documentation of all `QdrantVectorStore` features and configurations head to the API reference: https://python.langchain.com/api_reference/qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)