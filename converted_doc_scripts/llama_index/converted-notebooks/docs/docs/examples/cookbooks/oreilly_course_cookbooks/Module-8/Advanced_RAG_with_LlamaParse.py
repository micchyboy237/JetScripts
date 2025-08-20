from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.postprocessor.flag_embedding_reranker import (
FlagEmbeddingReranker,
)
from llama_parse import LlamaParse
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## Building Advanced RAG With LlamaParse

In this notebook we will demonstrate the following:

1. Using LlamaParse.
2. Using Recursive Retrieval with LlamaParse to query tables/ text within a document hierarchically.

[LlamaParse Documentation](https://github.com/run-llama/llama_parse/)

#### Installation
"""
logger.info("## Building Advanced RAG With LlamaParse")

# !pip install llama-index
# !pip install llama-index-postprocessor-flag-embedding-reranker
# !pip install git+https://github.com/FlagOpen/FlagEmbedding.git
# !pip install llama-parse

"""
#### Download Data
"""
logger.info("#### Download Data")

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_march_2022.pdf' -O './uber_10q_march_2022.pdf'

"""
#### Setting API Keys
"""
logger.info("#### Setting API Keys")

# import nest_asyncio

# nest_asyncio.apply()


os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..."

# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
#### Setting LLM and Embedding Model
"""
logger.info("#### Setting LLM and Embedding Model")


embed_model = MLXEmbedding(model="mxbai-embed-large")
llm = MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

Settings.llm = llm
Settings.embed_model = embed_model

"""
#### LlamaParse PDF reader for PDF Parsing

We compare two different retrieval/ queryengine strategies.

1. Using raw Markdown text as nodes for building index and applying a simple query engine for generating results.
2. Using MarkdownElementNodeParser for parsing the LlamaParse output Markdown results and building a recursive retriever query engine for generation.
"""
logger.info("#### LlamaParse PDF reader for PDF Parsing")


documents = LlamaParse(result_type="markdown").load_data(
    "./uber_10q_march_2022.pdf"
)

logger.debug(documents[0].text[:1000] + "...")


node_parser = MarkdownElementNodeParser(
    llm=MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats"), num_workers=8
)

nodes = node_parser.get_nodes_from_documents(documents)

text_nodes, index_nodes = node_parser.get_nodes_and_objects(nodes)

text_nodes[0]

index_nodes[0]

"""
#### Build Index
"""
logger.info("#### Build Index")

recursive_index = VectorStoreIndex(nodes=text_nodes + index_nodes)
raw_index = VectorStoreIndex.from_documents(documents)

"""
#### Create Query Engines
"""
logger.info("#### Create Query Engines")


reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
)

recursive_query_engine = recursive_index.as_query_engine(
    similarity_top_k=15, node_postprocessors=[reranker], verbose=True
)

raw_query_engine = raw_index.as_query_engine(
    similarity_top_k=15, node_postprocessors=[reranker]
)

"""
#### Querying with two different query engines

we compare base query engine vs recursive query engine with tables

##### Table Query Task: Queries for Table Question Answering
"""
logger.info("#### Querying with two different query engines")

query = "What is the change of free cash flow and what is the rate from the financial and operational highlights?"

response_1 = raw_query_engine.query(query)
logger.debug("\n************New LlamaParse+ Basic Query Engine************")
logger.debug(response_1)

response_2 = recursive_query_engine.query(query)
logger.debug(
    "\n************New LlamaParse+ Recursive Retriever Query Engine************"
)
logger.debug(response_2)

logger.info("\n\n[DONE]", bright=True)