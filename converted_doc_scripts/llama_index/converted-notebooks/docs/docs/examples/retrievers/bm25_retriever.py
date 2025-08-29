from jet.models.config import MODELS_CACHE_DIR
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.types import (
MetadataFilters,
MetadataFilter,
FilterOperator,
FilterCondition,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
import Stemmer
import chromadb
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/bm25_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# BM25 Retriever
In this guide, we define a bm25 retriever that search documents using the bm25 method.
BM25 (Best Matching 25) is a ranking function that extends TF-IDF by considering term frequency saturation and document length. BM25 effectively ranks documents based on query term occurrence and rarity across the corpus.

This notebook is very similar to the RouterQueryEngine notebook.

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# BM25 Retriever")

# %pip install llama-index
# %pip install llama-index-retrievers-bm25


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."


Settings.llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load Data

We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.
"""
logger.info("## Load Data")


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


splitter = SentenceSplitter(chunk_size=512)

nodes = splitter.get_nodes_from_documents(documents)

"""
## BM25 Retriever + Disk Persistence

One option is to create the `BM25Retriever` directly from nodes, and save to and from disk.
"""
logger.info("## BM25 Retriever + Disk Persistence")


bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=2,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)

bm25_retriever.persist("./bm25_retriever")

loaded_bm25_retriever = BM25Retriever.from_persist_dir("./bm25_retriever")

"""
## BM25 Retriever + Docstore Persistence

Here, we cover using a `BM25Retriever` with a docstore to hold your nodes. The advantage here is that the docstore can be remote (mongodb, redis, etc.)
"""
logger.info("## BM25 Retriever + Docstore Persistence")


docstore = SimpleDocumentStore()
docstore.add_documents(nodes)


bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=2,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)


retrieved_nodes = bm25_retriever.retrieve(
    "What happened at Viaweb and Interleaf?"
)
for node in retrieved_nodes:
    display_source_node(node, source_length=5000)

retrieved_nodes = bm25_retriever.retrieve("What did the author do after RISD?")
for node in retrieved_nodes:
    display_source_node(node, source_length=5000)

"""
## BM25 Retriever + MetadataFiltering
"""
logger.info("## BM25 Retriever + MetadataFiltering")


documents = [
    Document(text="Hello, world!", metadata={"key": "1"}),
    Document(text="Hello, world! 2", metadata={"key": "2"}),
    Document(text="Hello, world! 3", metadata={"key": "3"}),
    Document(text="Hello, world! 2.1", metadata={"key": "2"}),
]



splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)


filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="key",
            value="2",
            operator=FilterOperator.EQ,
        )
    ],
    condition=FilterCondition.AND,
)



retrieved_nodes = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=3,
    filters=filters,  # Add filters here
    stemmer=Stemmer.Stemmer("english"),
    language="english",
).retrieve("Hello, world!")

for node in retrieved_nodes:
    display_source_node(node, source_length=5000)

"""
## Hybrid Retriever with BM25 + Chroma

Now we will combine bm25 and chroma for sparse and dense retrieval.

The results are combined using the `QueryFusionRetriever`.

With the retriever, we can make a complete `RetrieverQueryEngine`.
"""
logger.info("## Hybrid Retriever with BM25 + Chroma")


docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("dense_vectors")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(
    docstore=docstore, vector_store=vector_store
)

index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

# import nest_asyncio

# nest_asyncio.apply()


retriever = QueryFusionRetriever(
    [
        index.as_retriever(similarity_top_k=2),
        BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=2
        ),
    ],
    num_queries=1,
    use_async=True,
)

nodes = retriever.retrieve("What happened at Viaweb and Interleaf?")
for node in nodes:
    display_source_node(node, source_length=5000)


query_engine = RetrieverQueryEngine(retriever)

response = query_engine.query("What did the author do after RISD?")
logger.debug(response)

"""
### Save and Load w/ a Vector Store

With our data in chroma, and our nodes in our docstore, we can save and recreate!

The vector store is already saved automatically by chroma, but we will need to save our docstore.
"""
logger.info("### Save and Load w/ a Vector Store")

storage_context.docstore.persist("./docstore.json")

"""
Now, we can reload and re-create our index.
"""
logger.info("Now, we can reload and re-create our index.")

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("dense_vectors")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

docstore = SimpleDocumentStore.from_persist_path("./docstore.json")

storage_context = StorageContext.from_defaults(
    docstore=docstore, vector_store=vector_store
)

index = VectorStoreIndex(nodes=[], storage_context=storage_context)

logger.info("\n\n[DONE]", bright=True)