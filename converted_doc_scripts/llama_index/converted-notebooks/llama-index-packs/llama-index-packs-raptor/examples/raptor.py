from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.packs.raptor import RaptorPack
from llama_index.packs.raptor import RaptorRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
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
# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

This notebook shows how to use an implementation of RAPTOR with llama-index, leveraging the RAPTOR llama-pack.

RAPTOR works by recursively clustering and summarizing clusters in layers for retrieval.

There two retrieval modes:
- tree_traversal -- traversing the tree of clusters, performing top-k at each level in the tree.
- collapsed -- treat the entire tree as a giant pile of nodes, perform simple top-k.

See [the paper](https://arxiv.org/abs/2401.18059) for full algorithm details.

## Setup
"""
logger.info("# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval")

# !pip install llama-index llama-index-packs-raptor llama-index-vector-stores-chroma
# !pip install --upgrade transformers


# !wget https://arxiv.org/pdf/2401.18059.pdf -O ./raptor_paper.pdf


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
## Constructing the Clusters/Hierarchy Tree
"""
logger.info("## Constructing the Clusters/Hierarchy Tree")

# import nest_asyncio

# nest_asyncio.apply()


documents = SimpleDirectoryReader(input_files=["./raptor_paper.pdf"]).load_data()


client = chromadb.PersistentClient(path="./raptor_paper_db")
collection = client.get_or_create_collection("raptor")

vector_store = ChromaVectorStore(chroma_collection=collection)

raptor_pack = RaptorPack(
    documents,
    embed_model=MLXEmbedding(
        model="mxbai-embed-large"
    ),  # used for embedding clusters
    llm=MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.1),  # used for generating summaries
    vector_store=vector_store,  # used for storage
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="collapsed",  # sets default mode
    transformations=[
        SentenceSplitter(chunk_size=400, chunk_overlap=50)
    ],  # transformations applied for ingestion
)

"""
## Retrieval
"""
logger.info("## Retrieval")

nodes = raptor_pack.run("What baselines is raptor compared against?", mode="collapsed")
logger.debug(len(nodes))
logger.debug(nodes[0].text)

nodes = raptor_pack.run(
    "What baselines is raptor compared against?", mode="tree_traversal"
)
logger.debug(len(nodes))
logger.debug(nodes[0].text)

"""
## Loading

Since we saved to a vector store, we can also use it again! (For local vector stores, there is a `persist` and `from_persist_dir` method on the retriever)
"""
logger.info("## Loading")


retriever = RaptorRetriever(
    [],
    embed_model=MLXEmbedding(
        model="mxbai-embed-large"
    ),  # used for embedding clusters
    llm=MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.1),  # used for generating summaries
    vector_store=vector_store,  # used for storage
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="tree_traversal",  # sets default mode
)

"""
## Query Engine
"""
logger.info("## Query Engine")


query_engine = RetrieverQueryEngine.from_args(
    retriever, llm=MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.1)
)

response = query_engine.query("What baselines was RAPTOR compared against?")

logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)