from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/reciprocal_rerank_fusion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Reciprocal Rerank Fusion Retriever
# 
# In this example, we walk through how you can combine retrieval results from multiple queries and multiple indexes. 
# 
# The retrieved nodes will be reranked according to the `Reciprocal Rerank Fusion` algorithm demonstrated in this [paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). It provides an effecient method for rerranking retrieval results without excessive computation or reliance on external models.
# 
# Full credits go to @Raduaschl on github for their [example implementation here](https://github.com/Raudaschl/rag-fusion).

# %pip install llama-index-llms-ollama
# %pip install llama-index-retrievers-bm25

import os
import openai

# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

## Setup

# 
# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# Download Data

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/retrievers/data/").load_data()

# Next, we will setup a vector index over the documentation.

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=256)

index = VectorStoreIndex.from_documents(documents, transformations=[splitter])

## Create a Hybrid Fusion Retriever
# 
# In this step, we fuse our index with a BM25 based retriever. This will enable us to capture both semantic relations and keywords in our input queries.
# 
# Since both of these retrievers calculate a score, we can use the reciprocal rerank algorithm to re-sort our nodes without using an additional models or excessive computation.
# 
# This setup will also query 4 times, once with your original query, and generate 3 more queries.
# 
# By default, it uses the following prompt to generate extra queries:
# 
# ```python
# QUERY_GEN_PROMPT = (
#     "You are a helpful assistant that generates multiple search queries based on a "
#     "single input query. Generate {num_queries} search queries, one on each line, "
#     "related to the following input query:\n"
#     "Query: {query}\n"
#     "Queries:\n"
# )
# ```

# First, we create our retrievers. Each will retrieve the top-2 most similar nodes:

from llama_index.retrievers.bm25 import BM25Retriever

vector_retriever = index.as_retriever(similarity_top_k=2)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=2
)

# Next, we can create our fusion retriever, which well return the top-2 most similar nodes from the 4 returned nodes from the retrievers:

from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=2,
    num_queries=4,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
)

import nest_asyncio

nest_asyncio.apply()

nodes_with_scores = retriever.retrieve(
    "What happened at Interleafe and Viaweb?"
)

for node in nodes_with_scores:
    print(f"Score: {node.score:.2f} - {node.text}...\n-----\n")

# As we can see, both retruned nodes correctly mention Viaweb and Interleaf!

## Use in a Query Engine!
# 
# Now, we can plug our retriever into a query engine to synthesize natural language responses.

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)

response = query_engine.query("What happened at Interleafe and Viaweb?")

from llama_index.core.response.notebook_utils import display_response

display_response(response)

logger.info("\n\n[DONE]", bright=True)