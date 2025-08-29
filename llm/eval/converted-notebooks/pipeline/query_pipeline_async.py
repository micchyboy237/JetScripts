import time
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import Document, VectorStoreIndex
from jet.llm.ollama.base import Ollama
from llama_index.core.llms import LLM
from llama_index.core.llama_pack import BaseLlamaPack
from typing import Dict, Any, List, Optional
from llama_index.core.query_pipeline import (
    QueryPipeline,
    InputComponent,
    ArgPackComponent,
)
from llama_index.core import SimpleDirectoryReader
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Query Pipeline with Async/Parallel Execution
#
# Here we showcase our query pipeline with async + parallel execution.
#
# We do this by setting up a RAG pipeline that does the following:
# 1. Send query to multiple RAG query engines.
# 2. Combine results.
#
# In the process we'll also show some nice abstractions for joining results (e.g. our `ArgPackComponent()`)

# Load Data
#
# Load in the Paul Graham essay as an example.

# %pip install llama-index-llms-ollama

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt' -O pg_essay.txt


reader = SimpleDirectoryReader(input_files=["pg_essay.txt"])
documents = reader.load_data()

# Setup Query Pipeline
#
# We setup a parallel query pipeline that executes multiple chunk sizes at once, and combines the results.

# Define Modules
#
# This includes:
# - LLM
# - Chunk Sizes
# - Query Engines


llm = Ollama(model="llama3.2")
chunk_sizes = [128, 256, 512, 1024]
query_engines = {}
for chunk_size in chunk_sizes:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)
    query_engines[str(chunk_size)] = vector_index.as_query_engine(llm=llm)

# Construct Query Pipeline
#
# Connect input to multiple query engines, and join the results.

p = QueryPipeline(verbose=True)
module_dict = {
    **query_engines,
    "input": InputComponent(),
    "summarizer": TreeSummarize(),
    "join": ArgPackComponent(
        convert_fn=lambda x: NodeWithScore(node=TextNode(text=str(x)))
    ),
}
p.add_modules(module_dict)
for chunk_size in chunk_sizes:
    p.add_link("input", str(chunk_size))
    p.add_link(str(chunk_size), "join", dest_key=str(chunk_size))
p.add_link("join", "summarizer", dest_key="nodes")
p.add_link("input", "summarizer", dest_key="query_str")

# Try out Queries
#
# Let's compare the async performance vs. synchronous performance!
#
# In our experiments we get a 2x speedup.


start_time = time.time()
response = await p.arun(input="What did the author do during his time in YC?")
print(str(response))
end_time = time.time()
print(f"Time taken: {end_time - start_time}")

start_time = time.time()
response = p.run(input="What did the author do during his time in YC?")
print(str(response))
end_time = time.time()
print(f"Time taken: {end_time - start_time}")

logger.info("\n\n[DONE]", bright=True)
