import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
import nest_asyncio
import os
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Property Graph Index
#
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_basic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
#
# In this notebook, we demonstrate some basic usage of the `PropertyGraphIndex` in LlamaIndex.
#
# The property graph index here will take unstructured documents, extract a property graph from it, and provide various methods to query that graph.

# %pip install llama-index

# Setup


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


nest_asyncio.apply()


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries/").load_data()

# Construction


index = PropertyGraphIndex.from_documents(
    documents,
    llm=Ollama(model="llama3.2", request_timeout=300.0,
               context_window=4096, temperature=0.3),
    embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
    show_progress=True,
)

# So lets recap what exactly just happened
# - `PropertyGraphIndex.from_documents()` - we loaded documents into an index
# - `Parsing nodes` - the index parsed the documents into nodes
# - `Extracting paths from text` - the nodes were passed to an LLM, and the LLM was prompted to generate knowledge graph triples (i.e. paths)
# - `Extracting implicit paths` - each `node.relationships` property was used to infer implicit paths
# - `Generating embeddings` - embeddings were generated for each text node and graph node (hence this happens twice)

# Lets explore what we created! For debugging purposes, the default `SimplePropertyGraphStore` includes a helper to save a `networkx` representation of the graph to an `html` file.

index.property_graph_store.save_networkx_graph(name="./kg.html")

# Opening the html in a browser, we can see our graph!
#
# If you zoom in, each "dense" node with many connections is actually the source chunk, with extracted entities and relations branching off from there.

# ![example graph](./kg_screenshot.png)

# Customizing Low-Level Construction
#
# If we wanted, we can do the same ingestion using the low-level API, leverage `kg_extractors`.


index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
    kg_extractors=[
        ImplicitPathExtractor(),
        SimpleLLMPathExtractor(
            llm=Ollama(model="llama3.2", request_timeout=300.0,
                       context_window=4096, temperature=0.3),
            num_workers=4,
            max_paths_per_chunk=10,
        ),
    ],
    show_progress=True,
)

# For a full guide on all extractors, see the [detailed usage page](/../../module_guides/indexing/lpg_index_guide#construction).

# Querying
#
# Querying a property graph index typically consists of using one or more sub-retrievers and combining results.
#
# Graph retrieval can be thought of
# - selecting node(s)
# - traversing from those nodes
#
# By default, two types of retrieval are used in unison
# - synoynm/keyword expansion - use the LLM to generate synonyms and keywords from the query
# - vector retrieval - use embeddings to find nodes in your graph
#
# Once nodes are found, you can either
# - return the paths adjacent to the selected nodes (i.e. triples)
# - return the paths + the original source text of the chunk (if available)

retriever = index.as_retriever(
    include_text=False,  # include source text, default True
)

nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

for node in nodes:
    print(node.text)

query_engine = index.as_query_engine(
    include_text=True,
)

response = query_engine.query("What happened at Interleaf and Viaweb?")

print(str(response))

# For full details on customizing retrieval and querying, see [the docs page](/../../module_guides/indexing/lpg_index_guide#retrieval-and-querying).

# Storage
#
# By default, storage happens using our simple in-memory abstractions - `SimpleVectorStore` for embeddings and `SimplePropertyGraphStore` for the property graph.
#
# We can save and load these to/from disk.

index.storage_context.persist(persist_dir="./storage")


index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage")
)

# Vector Stores
#
# While some graph databases support vectors (like Neo4j), you can still specify the vector store to use on top of your graph for cases where its not supported, or cases where you want to override.
#
# Below we will combine `ChromaVectorStore` with the default `SimplePropertyGraphStore`.

# %pip install llama-index-vector-stores-chroma


client = chromadb.PersistentClient("./chroma_db")
collection = client.get_or_create_collection("my_graph_vector_db")

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
    graph_store=SimplePropertyGraphStore(),
    vector_store=ChromaVectorStore(collection=collection),
    show_progress=True,
)

index.storage_context.persist(persist_dir="./storage")

# Then to load:

index = PropertyGraphIndex.from_existing(
    SimplePropertyGraphStore.from_persist_dir("./storage"),
    vector_store=ChromaVectorStore(chroma_collection=collection),
    llm=Ollama(model="llama3.2", request_timeout=300.0,
               context_window=4096, temperature=0.3),
)

# This looks slightly different than purely using the storage context, but the syntax is more concise now that we've started to mix things together.

logger.info("\n\n[DONE]", bright=True)
