"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/flare_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# FLARE Query Engine

Adapted from the paper "Active Retrieval Augmented Generation"

Currently implements FLARE Instruct, which tells the LLM to generate retrieval instructions.
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-llms-ollama

# !pip install llama-index


from llama_index.core.query_engine import FLAREInstructQueryEngine
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from jet.llm.ollama.base import Ollama
from llama_index.core import Settings
Settings.llm = Ollama(
    model="llama3.1", request_timeout=300.0, context_window=4096, temperature=0)
Settings.chunk_size = 512

"""
## Download Data
"""

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load Data
"""


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
)

index_query_engine = index.as_query_engine(similarity_top_k=2)


flare_query_engine = FLAREInstructQueryEngine(
    query_engine=index_query_engine,
    max_iterations=7,
    verbose=True,
)

response = flare_query_engine.query(
    "Can you tell me about the author's trajectory in the startup world?"
)

print(response)

response = flare_query_engine.query(
    "Can you tell me about what the author did during his time at YC?"
)

print(response)

response = flare_query_engine.query(
    "Tell me about the author's life from childhood to adulthood"
)

print(response)

response = index_query_engine.query(
    "Can you tell me about the author's trajectory in the startup world?"
)

print(str(response))

response = index_query_engine.query(
    "Tell me about the author's life from childhood to adulthood"
)

print(str(response))
