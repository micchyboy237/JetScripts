from jet.models.config import MODELS_CACHE_DIR
from IPython.display import Markdown, display
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
)
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.azure_openai import AzureOllamaFunctionCallingAdapter
from pyvis.network import Network
import json
import logging
import openai
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Nebula Graph Store
"""
logger.info("# Nebula Graph Store")

# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-graph-stores-nebula
# %pip install llama-index-llms-azure-openai


# os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

llm = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2")

Settings.llm = llm
Settings.chunk_size = 512


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_type = "azure"
openai.api_base = "https://<foo-bar>.openai.azure.com"
openai.api_version = "2022-12-01"
# os.environ["OPENAI_API_KEY"] = "<your-openai-key>"
# openai.api_key = os.getenv("OPENAI_API_KEY")

llm = AzureOllamaFunctionCallingAdapter(
    model="<foo-bar-model>",
    engine="<foo-bar-deployment>",
    temperature=0,
    api_key=openai.api_key,
    api_type=openai.api_type,
    api_base=openai.api_base,
    api_version=openai.api_version,
)

embedding_model = HuggingFaceEmbedding(
    model="text-embedding-ada-002",
    deployment_name="<foo-bar-deployment>",
    api_key=openai.api_key,
    api_base=openai.api_base,
    api_type=openai.api_type,
    api_version=openai.api_version,
)

Settings.llm = llm
Settings.chunk_size = chunk_size
Settings.embed_model = embedding_model

"""
## Using Knowledge Graph with NebulaGraphStore

#### Building the Knowledge Graph
"""
logger.info("## Using Knowledge Graph with NebulaGraphStore")


documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()

"""
## Prepare for NebulaGraph
"""
logger.info("## Prepare for NebulaGraph")

# %pip install nebula3-python

os.environ["NEBULA_USER"] = "root"
os.environ[
    "NEBULA_PASSWORD"
] = "<password>"  # replace with your password, by default it is "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"  # assumed we have NebulaGraph 3.5.0 or newer installed locally


space_name = "paul_graham_essay"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

"""
## Instantiate GPTNebulaGraph KG Indexes
"""
logger.info("## Instantiate GPTNebulaGraph KG Indexes")

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

"""
#### Querying the Knowledge Graph
"""
logger.info("#### Querying the Knowledge Graph")

query_engine = index.as_query_engine()

response = query_engine.query("Tell me more about Interleaf")

display(Markdown(f"<b>{response}</b>"))

response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf"
)

display(Markdown(f"<b>{response}</b>"))

"""
## Visualizing the Graph RAG

If we visualize the Graph based RAG, starting from the term `['Interleaf', 'history', 'Software', 'Company'] `, we could see how those connected context looks like, and it's a different form of Info./Knowledge:

- Refined and Concise Form
- Fine-grained Segmentation
- Interconnected-sturcutred nature
"""
logger.info("## Visualizing the Graph RAG")

# %pip install ipython-ngql networkx pyvis
# %load_ext ngql

# %ngql --address 127.0.0.1 --port 9669 --user root --password <password>

# %%ngql
USE paul_graham_essay
MATCH p = (n)-[*1..2]-()
WHERE id(n) IN['Interleaf', 'history', 'Software', 'Company']
RETURN p LIMIT 100

# %ng_draw

"""
#### Query with embeddings
"""
logger.info("#### Query with embeddings")

index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)

response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf"
)

display(Markdown(f"<b>{response}</b>"))

"""
#### Query with more global(cross node) context
"""
logger.info("#### Query with more global(cross node) context")

query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
    explore_global_knowledge=True,
)

response = query_engine.query("Tell me more about what the author and Lisp")

"""
#### Visualizing the Graph
"""
logger.info("#### Visualizing the Graph")


g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("example.html")

"""
#### [Optional] Try building the graph and manually add triplets!
"""
logger.info(
    "#### [Optional] Try building the graph and manually add triplets!")


node_parser = SentenceSplitter()

nodes = node_parser.get_nodes_from_documents(documents)

index = KnowledgeGraphIndex.from_documents([], storage_context=storage_context)

node_0_tups = [
    ("author", "worked on", "writing"),
    ("author", "worked on", "programming"),
]
for tup in node_0_tups:
    index.upsert_triplet_and_node(tup, nodes[0])

node_1_tups = [
    ("Interleaf", "made software for", "creating documents"),
    ("Interleaf", "added", "scripting language"),
    ("software", "generate", "web sites"),
]
for tup in node_1_tups:
    index.upsert_triplet_and_node(tup, nodes[1])

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)

response = query_engine.query("Tell me more about Interleaf")

str(response)

logger.info("\n\n[DONE]", bright=True)
