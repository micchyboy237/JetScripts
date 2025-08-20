from IPython.display import Markdown, display
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pyvis.network import Network
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Knowledge Graph Index

This tutorial gives a basic overview of how to use our `KnowledgeGraphIndex`, which handles
automated knowledge graph construction from unstructured text as well as entity-based querying.

If you would like to query knowledge graphs in more flexible ways, including pre-existing ones, please
check out our `KnowledgeGraphQueryEngine` and other constructs.
"""
logger.info("# Knowledge Graph Index")

# %pip install llama-index-llms-ollama


# os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

"""
## Using Knowledge Graph

#### Building the Knowledge Graph
"""
logger.info("## Using Knowledge Graph")



documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()

llm = MLX(temperature=0, model="text-davinci-002")
Settings.llm = llm
Settings.chunk_size = 512


graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
)

"""
#### [Optional] Try building the graph and manually add triplets!

#### Querying the Knowledge Graph
"""
logger.info("#### [Optional] Try building the graph and manually add triplets!")

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)

display(Markdown(f"<b>{response}</b>"))

query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf",
)

display(Markdown(f"<b>{response}</b>"))

"""
#### Query with embeddings
"""
logger.info("#### Query with embeddings")

new_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    include_embeddings=True,
)

query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf",
)

display(Markdown(f"<b>{response}</b>"))

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
logger.info("#### [Optional] Try building the graph and manually add triplets!")


node_parser = SentenceSplitter()

nodes = node_parser.get_nodes_from_documents(documents)

index = KnowledgeGraphIndex(
    [],
)

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
response = query_engine.query(
    "Tell me more about Interleaf",
)

str(response)

logger.info("\n\n[DONE]", bright=True)