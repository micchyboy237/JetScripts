from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Simple Fusion Retriever
# 
# In this example, we walk through how you can combine retrieval results from multiple queries and multiple indexes. 
# 
# The retrieved nodes will be returned as the top-k across all queries and indexes, as well as handling de-duplication of any nodes.

import os
import openai

# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

## Setup
# 
# For this notebook, we will use two very similar pages of our documentation, each stored in a separaete index.

from llama_index.core import SimpleDirectoryReader

documents_1 = SimpleDirectoryReader(
    input_files=["../../community/integrations/vector_stores.md"]
).load_data()
documents_2 = SimpleDirectoryReader(
    input_files=["../../module_guides/storing/vector_stores.md"]
).load_data()

from llama_index.core import VectorStoreIndex

index_1 = VectorStoreIndex.from_documents(documents_1)
index_2 = VectorStoreIndex.from_documents(documents_2)

## Fuse the Indexes!
# 
# In this step, we fuse our indexes into a single retriever. This retriever will also generate augment our query by generating extra queries related to the original question, and aggregate the results.
# 
# This setup will query 4 times, once with your original query, and generate 3 more queries.
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

from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [index_1.as_retriever(), index_2.as_retriever()],
    similarity_top_k=2,
    num_queries=4,  # set this to 1 to disable query generation
    use_async=True,
    verbose=True,
)

import nest_asyncio

nest_asyncio.apply()

nodes_with_scores = retriever.retrieve("How do I setup a chroma vector store?")

for node in nodes_with_scores:
    print(f"Score: {node.score:.2f} - {node.text[:100]}...")

## Use in a Query Engine!
# 
# Now, we can plug our retriever into a query engine to synthesize natural language responses.

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)

response = query_engine.query(
    "How do I setup a chroma vector store? Can you give an example?"
)

from llama_index.core.response.notebook_utils import display_response

display_response(response)

logger.info("\n\n[DONE]", bright=True)