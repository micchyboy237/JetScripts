from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# FalkorDB Property Graph Index
# 
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_falkordb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# 
# FalkorDB is a production-grade graph database, capable of storing a property graph, performing vector search, filtering, and more.
# 
# The easiest way to get started is with a cloud-hosted instance using [FalkorDB Cloud](https://app.falkordb.cloud)
# 
# For this notebook, we will instead cover how to run the database locally with docker.
# 
# If you already have an existing graph, please skip to the end of this notebook.

# %pip install llama-index llama-index-graph-stores-falkordb

## Docker Setup
# 
# To launch FalkorDB locally, first ensure you have docker installed. Then, you can launch the database with the following docker command
# 
# ```bash
# docker run \
#     -p 3000:3000 -p 6379:6379 \
#     -v $PWD/data:/data \
#     falkordb/falkordb:latest
# ```
# 
# From here, you can open the db at [http://localhost:3000/](http://localhost:3000/). On this page, you will be asked to sign in.
# 
# After this, you are ready to create your first property graph!

## Env Setup
# 
# We need just a few environment setups to get started.

import os

# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

import nest_asyncio

nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume/").load_data()

## Index Construction

from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore

graph_store = FalkorDBPropertyGraphStore(
    url="falkor://localhost:6379",
)

from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=Ollama(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.0)
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

# Now that the graph is created, we can explore it in the UI by visting [http://localhost:3000/](http://localhost:3000/). 
# 
# The easiest way to see the entire graph is to use a cypher command like `"match n=() return n"` at the top.
# 
# To delete an entire graph, a useful command is `"match n=() detach delete n"`.

## Querying and Retrieval

retriever = index.as_retriever(
    include_text=False,  # include source text in returned nodes, default True
)

nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

for node in nodes:
    print(node.text)

query_engine = index.as_query_engine(include_text=True)

response = query_engine.query("What happened at Interleaf and Viaweb?")

print(str(response))

## Loading from an existing Graph
# 
# If you have an existing graph (either created with LlamaIndex or otherwise), we can connect to and use it!
# 
# **NOTE:** If your graph was created outside of LlamaIndex, the most useful retrievers will be [text to cypher](/../../module_guides/indexing/lpg_index_guide#texttocypherretriever) or [cypher templates](/../../module_guides/indexing/lpg_index_guide#cyphertemplateretriever). Other retrievers rely on properties that LlamaIndex inserts.

from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

graph_store = FalkorDBPropertyGraphStore(
    url="falkor://localhost:6379",
)

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=Ollama(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.3),
    embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
)

# From here, we can still insert more documents!

from llama_index.core import Document

document = Document(text="LlamaIndex is great!")

index.insert(document)

nodes = index.as_retriever(include_text=False).retrieve("LlamaIndex")

print(nodes[0].text)

# For full details on construction, retrieval, querying of a property graph, see the [full docs page](/../../module_guides/indexing/lpg_index_guide).

logger.info("\n\n[DONE]", bright=True)