from IPython.display import Markdown, display
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.kuzu import KuzuGraphStore
from pyvis.network import Network
import kuzu
import os
import shutil


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
# Kùzu Graph Store

This notebook walks through configuring `Kùzu` to be the backend for graph storage in LlamaIndex.
"""
logger.info("# Kùzu Graph Store")

# %pip install llama-index
# %pip install llama-index-llms-ollama
# %pip install llama-index-graph-stores-kuzu
# %pip install pyvis


# os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"

"""
## Prepare for Kùzu
"""
logger.info("## Prepare for Kùzu")


shutil.rmtree("./test1", ignore_errors=True)
shutil.rmtree("./test2", ignore_errors=True)
shutil.rmtree("./test3", ignore_errors=True)


db = kuzu.Database("test1")

"""
## Using Knowledge Graph with KuzuGraphStore
"""
logger.info("## Using Knowledge Graph with KuzuGraphStore")


graph_store = KuzuGraphStore(db)

"""
#### Building the Knowledge Graph
"""
logger.info("#### Building the Knowledge Graph")


documents = SimpleDirectoryReader(
    "../../../examples/data/paul_graham"
).load_data()

llm = MLX(temperature=0, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
Settings.llm = llm
Settings.chunk_size = 512


storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
)

"""
#### Querying the Knowledge Graph

First, we can query and send only the triplets to the LLM.
"""
logger.info("#### Querying the Knowledge Graph")

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)

display(Markdown(f"<b>{response}</b>"))

"""
For more detailed answers, we can also send the text from where the retrieved tripets were extracted.
"""
logger.info("For more detailed answers, we can also send the text from where the retrieved tripets were extracted.")

query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)

display(Markdown(f"<b>{response}</b>"))

"""
#### Query with embeddings
"""
logger.info("#### Query with embeddings")

db = kuzu.Database("test2")
graph_store = KuzuGraphStore(db)
storage_context = StorageContext.from_defaults(graph_store=graph_store)
new_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
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
net.show("kuzugraph_draw.html")

"""
#### [Optional] Try building the graph and manually add triplets!
"""
logger.info("#### [Optional] Try building the graph and manually add triplets!")


node_parser = SentenceSplitter()

nodes = node_parser.get_nodes_from_documents(documents)

db = kuzu.Database("test3")
graph_store = KuzuGraphStore(db)
storage_context = StorageContext.from_defaults(graph_store=graph_store)
index = KnowledgeGraphIndex(
    [],
    storage_context=storage_context,
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