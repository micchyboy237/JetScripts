from llama_index.core.query_engine import RetrieverQueryEngine
from typing import Any, List
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle
from jet.llm.utils.llama_index_utils import display_jet_source_node
from typing import Optional
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from jet.llm.ollama.base import OllamaEmbedding
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader
from pathlib import Path
from llama_index.vector_stores.pinecone import PineconeVectorStore
import os
from pinecone import Pinecone, Index, ServerlessSpec
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/low_level/retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Building Retrieval from Scratch
#
# In this tutorial, we show you how to build a standard retriever against a vector database, that will fetch nodes via top-k similarity.
#
# We use Pinecone as the vector database. We load in nodes using our high-level ingestion abstractions (to see how to build this from scratch, see our previous tutorial!).
#
# We will show how to do the following:
# 1. How to generate a query embedding
# 2. How to query the vector database using different search modes (dense, sparse, hybrid)
# 3. How to parse results into a set of Nodes
# 4. How to put this in a custom retriever

# Setup
#
# We build an empty Pinecone Index, and define the necessary LlamaIndex wrappers/abstractions so that we can start loading data into Pinecone.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-readers-file pymupdf
# %pip install llama-index-vector-stores-pinecone
# %pip install llama-index-embeddings-ollama

# !pip install llama-index

# Build Pinecone Index


api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)

dataset_name = "quickstart"
if dataset_name not in pc.list_indexes().names():
    pc.create_index(
        dataset_name,
        dimension=1536,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(dataset_name)

pinecone_index.delete(deleteAll=True)

# Create PineconeVectorStore
#
# Simple wrapper abstraction to use in LlamaIndex. Wrap in StorageContext so we can easily load in Nodes.


vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Load Documents

# !mkdir data
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"


loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

# Load into Vector Store
#
# Load in documents into the PineconeVectorStore.
#
# **NOTE**: We use high-level ingestion abstractions here, with `VectorStoreIndex.from_documents.` We'll refrain from using `VectorStoreIndex` for the rest of this tutorial.


splitter = SentenceSplitter(chunk_size=1024)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], storage_context=storage_context
)

# Define Vector Retriever
#
# Now we're ready to define our retriever against this vector store to retrieve a set of nodes.
#
# We'll show the processes step by step and then wrap it into a function.

query_str = "Can you tell me about the key concepts for safety finetuning"

# 1. Generate a Query Embedding


embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

query_embedding = embed_model.get_query_embedding(query_str)

# 2. Query the Vector Database
#
# We show how to query the vector database with different modes: default, sparse, and hybrid.
#
# We first construct a `VectorStoreQuery` and then query the vector db.


query_mode = "default"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)

query_result = vector_store.query(vector_store_query)
query_result

# 3. Parse Result into a set of Nodes
#
# The `VectorStoreQueryResult` returns the set of nodes and similarities. We construct a `NodeWithScore` object with this.


nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))


for node in nodes_with_scores:
    display_jet_source_node(node, source_length=1000)

# 4. Put this into a Retriever
#
# Let's put this into a Retriever subclass that can plug into the rest of LlamaIndex workflows!


class PineconeRetriever(BaseRetriever):
    """Retriever over a pinecone vector store."""

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        if query_bundle.embedding is None:
            query_embedding = self._embed_model.get_query_embedding(
                query_bundle.query_str
            )
        else:
            query_embedding = query_bundle.embedding

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


retriever = PineconeRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)

retrieved_nodes = retriever.retrieve(query_str)
for node in retrieved_nodes:
    display_jet_source_node(node, source_length=1000)

# Plug this into our RetrieverQueryEngine to synthesize a response
#
# **NOTE**: We'll cover more on how to build response synthesis from scratch in future tutorials!


query_engine = RetrieverQueryEngine.from_args(retriever)

response = query_engine.query(query_str)

print(str(response))

logger.info("\n\n[DONE]", bright=True)
