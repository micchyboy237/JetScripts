from IPython.display import Markdown, display
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
KnowledgeGraphIndex,
)
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.azure_openai import AzureMLX
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

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Neo4j Graph Store
"""
logger.info("# Neo4j Graph Store")

# %pip install llama-index-llms-ollama
# %pip install llama-index-graph-stores-neo4j
# %pip install llama-index-embeddings-ollama
# %pip install llama-index-llms-azure-openai


# os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

llm = MLX(temperature=0, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
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

llm = AzureMLX(
    deployment_name="<foo-bar-deployment>",
    temperature=0,
    openai_api_version=openai.api_version,
    model_kwargs={
        "api_key": openai.api_key,
        "api_base": openai.api_base,
        "api_type": openai.api_type,
        "api_version": openai.api_version,
    },
)

embedding_llm = MLXEmbedding(
    model="text-embedding-ada-002",
    deployment_name="<foo-bar-deployment>",
    api_key=openai.api_key,
    api_base=openai.api_base,
    api_type=openai.api_type,
    api_version=openai.api_version,
)

Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = 512

"""
## Using Knowledge Graph with Neo4jGraphStore

#### Building the Knowledge Graph
"""
logger.info("## Using Knowledge Graph with Neo4jGraphStore")




documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()

"""
## Prepare for Neo4j
"""
logger.info("## Prepare for Neo4j")

# %pip install neo4j

username = "neo4j"
password = "retractor-knot-thermocouples"
url = "bolt://44.211.44.239:7687"
database = "neo4j"

"""
## Instantiate Neo4jGraph KG Indexes
"""
logger.info("## Instantiate Neo4jGraph KG Indexes")

graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
)

"""
#### Querying the Knowledge Graph

First, we can query and send only the triplets to the LLM.
"""
logger.info("#### Querying the Knowledge Graph")

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)

response = query_engine.query("Tell me more about Interleaf")

display(Markdown(f"<b>{response}</b>"))

"""
For more detailed answers, we can also send the text from where the retrieved tripets were extracted.
"""
logger.info("For more detailed answers, we can also send the text from where the retrieved tripets were extracted.")

query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf"
)

display(Markdown(f"<b>{response}</b>"))

"""
#### Query with embeddings
"""
logger.info("#### Query with embeddings")

graph_store.query(
    """
MATCH (n) DETACH DELETE n
"""
)

index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
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
    "Tell me more about what the author worked on at Interleaf"
)

display(Markdown(f"<b>{response}</b>"))

"""
#### [Optional] Try building the graph and manually add triplets!
"""
logger.info("#### [Optional] Try building the graph and manually add triplets!")


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

display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)