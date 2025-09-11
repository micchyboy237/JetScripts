from jet.logger import logger
from langchain_community.retrievers import (
PineconeHybridSearchRetriever,
)
from langchain_pinecone import PineconeSparseVectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone.embeddings import PineconeSparseEmbeddings
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
keywords: [pinecone]
---

# Pinecone

>[Pinecone](https://docs.pinecone.io/docs/overview) is a vector database with broad functionality.


## Installation and Setup

Install the Python SDK:
"""
logger.info("# Pinecone")

pip install langchain-pinecone

"""
## Vector store

There exists a wrapper around Pinecone indexes, allowing you to use it as a vectorstore,
whether for semantic search or example selection.
"""
logger.info("## Vector store")


"""
For a more detailed walkthrough of the Pinecone vectorstore, see [this notebook](/docs/integrations/vectorstores/pinecone)

### Sparse Vector store

LangChain's `PineconeSparseVectorStore` enables sparse retrieval using Pinecone's sparse English model. It maps text to sparse vectors and supports adding documents and similarity search.
"""
logger.info("### Sparse Vector store")


vector_store = PineconeSparseVectorStore(
    index=my_index,
    embedding_model="pinecone-sparse-english-v0"
)
vector_store.add_documents(documents)
results = vector_store.similarity_search("your query", k=3)

"""
For a more detailed walkthrough, see the [Pinecone Sparse Vector Store notebook](/docs/integrations/vectorstores/pinecone_sparse).

### Sparse Embedding

LangChain's `PineconeSparseEmbeddings` provides sparse embedding generation using Pinecone's `pinecone-sparse-english-v0` model.
"""
logger.info("### Sparse Embedding")


sparse_embeddings = PineconeSparseEmbeddings(
    model="pinecone-sparse-english-v0"
)
query_embedding = sparse_embeddings.embed_query("sample text")

docs = ["Document 1 content", "Document 2 content"]
doc_embeddings = sparse_embeddings.embed_documents(docs)

"""
For more detailed usage, see the [Pinecone Sparse Embeddings notebook](/docs/integrations/vectorstores/pinecone_sparse).


## Retrievers

### Pinecone Hybrid Search
"""
logger.info("## Retrievers")

pip install pinecone pinecone-text

"""

"""


"""
For more detailed information, see [this notebook](/docs/integrations/retrievers/pinecone_hybrid_search).


### Self Query retriever

Pinecone vector store can be used as a retriever for self-querying.

For more detailed information, see [this notebook](/docs/integrations/retrievers/self_query/pinecone).
"""
logger.info("### Self Query retriever")

logger.info("\n\n[DONE]", bright=True)