from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import QueryBundle
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.postgres import PGVectorStore
from pathlib import Path
from sqlalchemy import make_url
from typing import Any, List
from typing import Optional
import os
import psycopg2
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/low_level/oss_ingestion_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Building RAG from Scratch (Open-source only!) 

In this tutorial, we show you how to build a data ingestion pipeline into a vector database, and then build a retrieval pipeline from that vector database, from scratch.

Notably, we use a fully open-source stack:

- Sentence Transformers as the embedding model
- Postgres as the vector store (we support many other [vector stores](https://gpt-index.readthedocs.io/en/stable/module_guides/storing/vector_stores.html) too!)
- Llama 2 as the LLM (through [llama.cpp](https://github.com/ggerganov/llama.cpp))

## Setup

We setup our open-source components.
1. Sentence Transformers
2. Llama 2
3. We initialize postgres and wrap it with our wrappers/abstractions.

#### Sentence Transformers
"""
logger.info("# Building RAG from Scratch (Open-source only!)")

# %pip install llama-index-readers-file pymupdf
# %pip install llama-index-vector-stores-postgres
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-llms-llama-cpp


embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

"""
#### Llama CPP

In this notebook, we use the [`llama-2-chat-13b-ggml`](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML) model, along with the proper prompt formatting. 

Check out our [Llama CPP guide](https://gpt-index.readthedocs.io/en/stable/examples/llm/llama_2_llama_cpp.html) for full setup instructions/details.
"""
logger.info("#### Llama CPP")

# !pip install llama-cpp-python


model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

llm = LlamaCPP(
    model_url=model_url,
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

"""
#### Initialize Postgres

Using an existing postgres running at localhost, create the database we'll be using.

**NOTE**: Of course there are plenty of other open-source/self-hosted databases you can use! e.g. Chroma, Qdrant, Weaviate, and many more. Take a look at our [vector store guide](https://gpt-index.readthedocs.io/en/stable/module_guides/storing/vector_stores.html).

**NOTE**: You will need to setup postgres on your local system. Here's an example of how to set it up on OSX: https://www.sqlshack.com/setting-up-a-postgresql-database-on-mac/.

**NOTE**: You will also need to install pgvector (https://github.com/pgvector/pgvector).

You can add a role like the following:
```
CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';
ALTER ROLE <user> SUPERUSER;
```
"""
logger.info("#### Initialize Postgres")

# !pip install psycopg2-binary pgvector asyncpg "sqlalchemy[asyncio]" greenlet


db_name = "vector_db"
host = "localhost"
password = "password"
port = "5432"
user = "jerry"
conn = psycopg2.connect(
    dbname="postgres",
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")


vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="llama2_paper",
    embed_dim=384,  # openai embedding dimension
)

"""
## Build an Ingestion Pipeline from Scratch

We show how to build an ingestion pipeline as mentioned in the introduction.

We fast-track the steps here (can skip metadata extraction). More details can be found [in our dedicated ingestion guide](https://gpt-index.readthedocs.io/en/latest/examples/low_level/ingestion.html).

### 1. Load Data
"""
logger.info("## Build an Ingestion Pipeline from Scratch")

# !mkdir data
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"

# from llama_index.readers.file import PyMuPDFReader

# loader = PyMuPDFReader()
documents = loader.load(
    file_path="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/llama2.pdf")

"""
### 2. Use a Text Splitter to Split Documents
"""
logger.info("### 2. Use a Text Splitter to Split Documents")


text_parser = SentenceSplitter(
    chunk_size=1024,
)

text_chunks = []
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

"""
### 3. Manually Construct Nodes from Text Chunks
"""
logger.info("### 3. Manually Construct Nodes from Text Chunks")


nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

"""
### 4. Generate Embeddings for each Node

Here we generate embeddings for each Node using a sentence_transformers model.
"""
logger.info("### 4. Generate Embeddings for each Node")

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

"""
### 5. Load Nodes into a Vector Store

We now insert these nodes into our `PostgresVectorStore`.
"""
logger.info("### 5. Load Nodes into a Vector Store")

vector_store.add(nodes)

"""
## Build Retrieval Pipeline from Scratch

We show how to build a retrieval pipeline. Similar to ingestion, we fast-track the steps. Take a look at our [retrieval guide](https://gpt-index.readthedocs.io/en/latest/examples/low_level/retrieval.html) for more details!
"""
logger.info("## Build Retrieval Pipeline from Scratch")

query_str = "Can you tell me about the key concepts for safety finetuning"

"""
### 1. Generate a Query Embedding
"""
logger.info("### 1. Generate a Query Embedding")

query_embedding = embed_model.get_query_embedding(query_str)

"""
### 2. Query the Vector Database
"""
logger.info("### 2. Query the Vector Database")


query_mode = "default"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)

query_result = vector_store.query(vector_store_query)
logger.debug(query_result.nodes[0].get_content())

"""
### 3. Parse Result into a Set of Nodes
"""
logger.info("### 3. Parse Result into a Set of Nodes")


nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))

"""
### 4. Put into a Retriever
"""
logger.info("### 4. Put into a Retriever")


class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
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
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


retriever = VectorDBRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)

"""
## Plug this into our RetrieverQueryEngine to synthesize a response
"""
logger.info("## Plug this into our RetrieverQueryEngine to synthesize a response")


query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

query_str = "How does Llama 2 perform compared to other open-source models?"

response = query_engine.query(query_str)

logger.debug(str(response))

logger.debug(response.source_nodes[0].get_content())

logger.info("\n\n[DONE]", bright=True)
