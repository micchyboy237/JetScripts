from jet.logger import logger
from langchain_core.documents import Document
from langchain_pinecone import PineconeSparseVectorStore
from langchain_pinecone.embeddings import PineconeSparseEmbeddings
from pinecone import AwsRegion, CloudProvider, Metric, ServerlessSpec
from pinecone import Pinecone
from uuid import uuid4
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
# Pinecone (sparse)

>[Pinecone](https://docs.pinecone.io/docs/overview) is a vector database with broad functionality.

This notebook shows how to use functionality related to the `Pinecone` vector database.

## Setup

To use the `PineconeSparseVectorStore` you first need to install the partner package, as well as the other packages used throughout this notebook.
"""
logger.info("# Pinecone (sparse)")

# %pip install -qU "langchain-pinecone==0.2.5"

"""
### Credentials
Create a new Pinecone account, or sign into your existing one, and create an API key to use in this notebook.
"""
logger.info("### Credentials")

# from getpass import getpass


# os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY") or getpass(
    "Enter your Pinecone API key: "
)

pc = Pinecone()

"""
## Initialization
Before initializing our vector store, let's connect to a Pinecone index. If one named index_name doesn't exist, it will be created.
"""
logger.info("## Initialization")


index_name = "langchain-sparse-vector-search"  # change if desired
model_name = "pinecone-sparse-english-v0"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1,
        embed={
            "model": model_name,
            "field_map": {"text": "chunk_text"},
            "metric": Metric.DOTPRODUCT,
        },
    )

index = pc.Index(index_name)
logger.debug(f"Index `{index_name}` host: {index.config.host}")

"""
For our sparse embedding model we use [`pinecone-sparse-english-v0`](https://docs.pinecone.io/models/pinecone-sparse-english-v0), we initialize it like so:
"""
logger.info("For our sparse embedding model we use [`pinecone-sparse-english-v0`](https://docs.pinecone.io/models/pinecone-sparse-english-v0), we initialize it like so:")


sparse_embeddings = PineconeSparseEmbeddings(model=model_name)

"""
Now that our Pinecone index and embedding model are both ready, we can initialize our sparse vector store in LangChain:
"""
logger.info("Now that our Pinecone index and embedding model are both ready, we can initialize our sparse vector store in LangChain:")


vector_store = PineconeSparseVectorStore(index=index, embedding=sparse_embeddings)

"""
## Manage vector store
Once you have created your vector store, we can interact with it by adding and deleting different items.

### Add items to vector store
We can add items to our vector store by using the `add_documents` function.
"""
logger.info("## Manage vector store")



documents = [
    Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "social"},
    ),
    Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "social"},
    ),
    Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "social"},
    ),
    Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
    ),
    Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
    ),
    Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "social"},
    ),
    Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "social"},
    ),
]

uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)

"""
### Delete items from vector store

We can delete records from our vector store using the `delete` method, providing it with a list of document IDs to delete.
"""
logger.info("### Delete items from vector store")

vector_store.delete(ids=[uuids[-1]])

"""
## Query vector store

Once we have loaded our documents into the vector store we're most likely ready to begin querying. There are various method for doing this in LangChain.

First, we'll see how to perform a simple vector search by querying our `vector_store` directly via the `similarity_search` method:
"""
logger.info("## Query vector store")

results = vector_store.similarity_search("I'm building a new LangChain project!", k=3)

for res in results:
    logger.debug(f"* {res.page_content} [{res.metadata}]")

"""
We can also add [metadata filtering](https://docs.pinecone.io/guides/data/understanding-metadata#metadata-query-language) to our query to limit our search based on various criteria. Let's try a simple filter to limit our search to include only records with `source=="social"`:
"""
logger.info("We can also add [metadata filtering](https://docs.pinecone.io/guides/data/understanding-metadata#metadata-query-language) to our query to limit our search based on various criteria. Let's try a simple filter to limit our search to include only records with `source=="social"`:")

results = vector_store.similarity_search(
    "I'm building a new LangChain project!",
    k=3,
    filter={"source": "social"},
)
for res in results:
    logger.debug(f"* {res.page_content} [{res.metadata}]")

"""
When comparing these results, we can see that our first query returned a different record from the `"website"` source. In our latter, filtered, query — this is no longer the case.

### Similarity Search and Scores

We can also search while returning the similarity score in a list of `(document, score)` tuples. Where the `document` is a LangChain `Document` object containing our text content and metadata.
"""
logger.info("### Similarity Search and Scores")

results = vector_store.similarity_search_with_score(
    "I'm building a new LangChain project!", k=3, filter={"source": "social"}
)
for doc, score in results:
    logger.debug(f"[SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

"""
### As a Retriever

In our chains and agents we'll often use the vector store as a `VectorStoreRetriever`. To create that, we use the `as_retriever` method:
"""
logger.info("### As a Retriever")

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5},
)
retriever

"""
We can now query our retriever using the `invoke` method:
"""
logger.info("We can now query our retriever using the `invoke` method:")

retriever.invoke(
    input="I'm building a new LangChain project!", filter={"source": "social"}
)

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

## API reference

For detailed documentation of all features and configurations head to the API reference: 
https://python.langchain.com/api_reference/pinecone/vectorstores_sparse/langchain_pinecone.vectorstores_sparse.PineconeSparseVectorStore.html#langchain_pinecone.vectorstores_sparse.PineconeSparseVectorStore

Sparse Embeddings:
https://python.langchain.com/api_reference/pinecone/embeddings/langchain_pinecone.embeddings.PineconeSparseEmbeddings.html
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)