from dataclasses import fields
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex
from llama_index.core.bridge.pydantic import Field
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import (
VectorStoreQuery,
VectorStoreQueryResult,
)
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
from typing import List, Any, Optional, Dict
from typing import Tuple
from typing import cast
import numpy as np
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/low_level/vector_store.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Building a (Very Simple) Vector Store from Scratch

In this tutorial, we show you how to build a simple in-memory vector store that can store documents along with metadata. It will also expose a query interface that can support a variety of queries:
- semantic search (with embedding similarity)
- metadata filtering

**NOTE**: Obviously this is not supposed to be a replacement for any actual vector store (e.g. Pinecone, Weaviate, Chroma, Qdrant, Milvus, or others within our wide range of vector store integrations). This is more to teach some key retrieval concepts, like top-k embedding search + metadata filtering.

We won't be covering advanced query/retrieval concepts such as approximate nearest neighbors, sparse/hybrid search, or any of the system concepts that would be required for building an actual database.

## Setup

We load in some documents, and parse them into Node objects - chunks that are ready to be inserted into a vector store.

#### Load in Documents
"""
logger.info("# Building a (Very Simple) Vector Store from Scratch")

# %pip install llama-index-readers-file pymupdf
# %pip install llama-index-embeddings-ollama

# !mkdir data
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O f"{GENERATED_DIR}/llama2.pdf"

# from llama_index.readers.file import PyMuPDFReader

# loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

"""
#### Parse into Nodes
"""
logger.info("#### Parse into Nodes")


node_parser = SentenceSplitter(chunk_size=256)
nodes = node_parser.get_nodes_from_documents(documents)

"""
#### Generate Embeddings for each Node
"""
logger.info("#### Generate Embeddings for each Node")


embed_model = MLXEmbedding()
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

"""
## Build a Simple In-Memory Vector Store

Now we'll build our in-memory vector store. We'll store Nodes within a simple Python dictionary. We'll start off implementing embedding search, and add metadata filters.

### 1. Defining the Interface

We'll first define the interface for building a vector store. It contains the following items:

- `get`
- `add`
- `delete`
- `query`
- `persist` (which we will not implement)
"""
logger.info("## Build a Simple In-Memory Vector Store")



class BaseVectorStore(BasePydanticVectorStore):
    """Simple custom Vector Store.

    Stores documents in a simple in-memory dict.

    """

    stores_text: bool = True

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        pass

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """Add nodes to index."""
        pass

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        pass

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        pass

    def persist(self, persist_path, fs=None) -> None:
        """Persist the SimpleVectorStore to a directory.

        NOTE: we are not implementing this for now.

        """
        pass

"""
At a high-level, we subclass our base `VectorStore` abstraction. There's no inherent reason to do this if you're just building a vector store from scratch. We do it because it makes it easy to plug into our downstream abstractions later.

Let's look at some of the classes defined here.
- `BaseNode` is simply the parent class of our core Node modules. Each Node represents a text chunk + associated metadata.
- We also use some lower-level constructs, for instance our `VectorStoreQuery` and `VectorStoreQueryResult`. These are just lightweight dataclass containers to represent queries and results. We look at the dataclass fields below.
"""
logger.info("At a high-level, we subclass our base `VectorStore` abstraction. There's no inherent reason to do this if you're just building a vector store from scratch. We do it because it makes it easy to plug into our downstream abstractions later.")


{f.name: f.type for f in fields(VectorStoreQuery)}

{f.name: f.type for f in fields(VectorStoreQueryResult)}

"""
### 2. Defining `add`, `get`, and `delete`

We add some basic capabilities to add, get, and delete from a vector store.

The implementation is very simple (everything is just stored in a python dictionary).
"""
logger.info("### 2. Defining `add`, `get`, and `delete`")



class VectorStore2(BaseVectorStore):
    """VectorStore2 (add/get/delete implemented)."""

    stores_text: bool = True
    node_dict: Dict[str, BaseNode] = Field(default_factory=dict)

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        return self.node_dict[text_id]

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """Add nodes to index."""
        for node in nodes:
            self.node_dict[node.node_id] = node

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with node_id.

        Args:
            node_id: str

        """
        del self.node_dict[node_id]

"""
We run some basic tests just to show it works well.
"""
logger.info("We run some basic tests just to show it works well.")

test_node = TextNode(id_="id1", text="hello world")
test_node2 = TextNode(id_="id2", text="foo bar")
test_nodes = [test_node, test_node2]

vector_store = VectorStore2()

vector_store.add(test_nodes)

node = vector_store.get("id1")
logger.debug(str(node))

"""
### 3.a Defining `query` (semantic search)

We implement a basic version of top-k semantic search. This simply iterates through all document embeddings, and compute cosine-similarity with the query embedding. The top-k documents by cosine similarity are returned.

Cosine similarity: $\dfrac{\vec{d}\vec{q}}{|\vec{d}||\vec{q}|}$ for every document, query embedding pair $\vec{d}$, $\vec{p}$.

**NOTE**: The top-k value is contained in the `VectorStoreQuery` container.

**NOTE**: Similar to the above, we define another subclass just so we don't have to reimplement the above functions (not because this is actually good code practice).
"""
logger.info("### 3.a Defining `query` (semantic search)")



def get_top_k_embeddings(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
    doc_ids: List[str],
    similarity_top_k: int = 5,
) -> Tuple[List[float], List]:
    """Get top nodes by similarity to the query."""
    qembed_np = np.array(query_embedding)
    dembed_np = np.array(doc_embeddings)
    dproduct_arr = np.dot(dembed_np, qembed_np)
    norm_arr = np.linalg.norm(qembed_np) * np.linalg.norm(
        dembed_np, axis=1, keepdims=False
    )
    cos_sim_arr = dproduct_arr / norm_arr

    tups = [(cos_sim_arr[i], doc_ids[i]) for i in range(len(doc_ids))]
    sorted_tups = sorted(tups, key=lambda t: t[0], reverse=True)

    sorted_tups = sorted_tups[:similarity_top_k]

    result_similarities = [s for s, _ in sorted_tups]
    result_ids = [n for _, n in sorted_tups]
    return result_similarities, result_ids



class VectorStore3A(VectorStore2):
    """Implements semantic/dense search."""

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""

        query_embedding = cast(List[float], query.query_embedding)
        doc_embeddings = [n.embedding for n in self.node_dict.values()]
        doc_ids = [n.node_id for n in self.node_dict.values()]

        similarities, node_ids = get_top_k_embeddings(
            query_embedding,
            doc_embeddings,
            doc_ids,
            similarity_top_k=query.similarity_top_k,
        )
        result_nodes = [self.node_dict[node_id] for node_id in node_ids]

        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )

"""
### 3.b. Supporting Metadata Filtering

The next extension is adding metadata filter support. This means that we will first filter the candidate set with documents that pass the metadata filters, and then perform semantic querying.

For simplicity we use metadata filters for exact matching with an AND condition.
"""
logger.info("### 3.b. Supporting Metadata Filtering")



def filter_nodes(nodes: List[BaseNode], filters: MetadataFilters):
    filtered_nodes = []
    for node in nodes:
        matches = True
        for f in filters.filters:
            if f.key not in node.metadata:
                matches = False
                continue
            if f.value != node.metadata[f.key]:
                matches = False
                continue
        if matches:
            filtered_nodes.append(node)
    return filtered_nodes

"""
We add `filter_nodes` as a first-pass over the nodes before running semantic search.
"""
logger.info("We add `filter_nodes` as a first-pass over the nodes before running semantic search.")

def dense_search(query: VectorStoreQuery, nodes: List[BaseNode]):
    """Dense search."""
    query_embedding = cast(List[float], query.query_embedding)
    doc_embeddings = [n.embedding for n in nodes]
    doc_ids = [n.node_id for n in nodes]
    return get_top_k_embeddings(
        query_embedding,
        doc_embeddings,
        doc_ids,
        similarity_top_k=query.similarity_top_k,
    )


class VectorStore3B(VectorStore2):
    """Implements Metadata Filtering."""

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        nodes = self.node_dict.values()
        if query.filters is not None:
            nodes = filter_nodes(nodes, query.filters)
        if len(nodes) == 0:
            result_nodes = []
            similarities = []
            node_ids = []
        else:
            similarities, node_ids = dense_search(query, nodes)
            result_nodes = [self.node_dict[node_id] for node_id in node_ids]
        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )

"""
### 4. Load Data into our Vector Store

Let's load our text chunks into the vector store, and run it on different types of queries: dense search, w/ metadata filters, and more.
"""
logger.info("### 4. Load Data into our Vector Store")

vector_store = VectorStore3B()
vector_store.add(nodes)

"""
Define an example question and embed it.
"""
logger.info("Define an example question and embed it.")

query_str = "Can you tell me about the key concepts for safety finetuning"
query_embedding = embed_model.get_query_embedding(query_str)

"""
#### Query the vector store with dense search.
"""
logger.info("#### Query the vector store with dense search.")

query_obj = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2
)

query_result = vector_store.query(query_obj)
for similarity, node in zip(query_result.similarities, query_result.nodes):
    logger.debug(
        "\n----------------\n"
        f"[Node ID {node.node_id}] Similarity: {similarity}\n\n"
        f"{node.get_content(metadata_mode='all')}"
        "\n----------------\n\n"
    )

"""
#### Query the vector store with dense search + Metadata Filters
"""
logger.info("#### Query the vector store with dense search + Metadata Filters")

filters = MetadataFilters.from_dict({"source": "24"})

query_obj = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, filters=filters
)

query_result = vector_store.query(query_obj)
for similarity, node in zip(query_result.similarities, query_result.nodes):
    logger.debug(
        "\n----------------\n"
        f"[Node ID {node.node_id}] Similarity: {similarity}\n\n"
        f"{node.get_content(metadata_mode='all')}"
        "\n----------------\n\n"
    )

"""
## Build a RAG System with the Vector Store

Now that we've built the RAG system, it's time to plug it into our downstream system!
"""
logger.info("## Build a RAG System with the Vector Store")


index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine()

query_str = "Can you tell me about the key concepts for safety finetuning"

response = query_engine.query(query_str)

logger.debug(str(response))

"""
## Conclusion

That's it! We've built a simple in-memory vector store that supports very simple inserts, gets, deletes, and supports dense search and metadata filtering. This can then be plugged into the rest of LlamaIndex abstractions.

It doesn't support sparse search yet and is obviously not meant to be used in any sort of actual app. But this should expose some of what's going on under the hood!
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)