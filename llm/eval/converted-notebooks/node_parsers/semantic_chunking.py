from jet.llm.utils.llama_index_utils import display_jet_source_node
from llama_index.core import VectorStoreIndex
import os
from jet.llm.ollama.base import OllamaEmbedding
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.core import SimpleDirectoryReader
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_parsers/semantic_chunking.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Semantic Chunker
#
# "Semantic chunking" is a new concept proposed Greg Kamradt in his video tutorial on 5 levels of embedding chunking: https://youtu.be/8OJC21T2SL4?t=1933.
#
# Instead of chunking text with a **fixed** chunk size, the semantic splitter adaptively picks the breakpoint in-between sentences using embedding similarity. This ensures that a "chunk" contains sentences that are semantically related to each other.
#
# We adapted it into a LlamaIndex module.
#
# Check out our notebook below!
#
# Caveats:
#
# - The regex primarily works for English sentences
# - You may have to tune the breakpoint percentile threshold.

# Setup Data

# %pip install llama-index-embeddings-ollama

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'pg_essay.txt'


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data",
    required_exts=[".md"]
).load_data()

# Define Semantic Splitter


# os.environ["OPENAI_API_KEY"] = "sk-..."

embed_model = OllamaEmbedding(model_name="nomic-embed-text")
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

base_splitter = SentenceSplitter(chunk_size=512)

nodes = splitter.get_nodes_from_documents(documents)

# Inspecting the Chunks
#
# Let's take a look at chunks produced by the semantic splitter.
#
# Chunk 1: IBM 1401

logger.newline()
logger.info("Semantic Chunk 1:")
logger.debug(nodes[1].get_content())

# Chunk 2: Personal Computer + College

logger.newline()
logger.info("Semantic Chunk 2:")
logger.debug(nodes[2].get_content())

# Chunk 3: Finishing up College + Grad School

logger.newline()
logger.info("Semantic Chunk 3:")
logger.debug(nodes[3].get_content())

# Compare against Baseline
#
# In contrast let's compare against the baseline with a fixed chunk size.

base_nodes = base_splitter.get_nodes_from_documents(documents)


logger.newline()
logger.info("Plain Chunk 1:")
logger.debug(base_nodes[1].get_content())

logger.newline()
logger.info("Plain Chunk 2:")
logger.debug(base_nodes[2].get_content())

logger.newline()
logger.info("Plain Chunk 3:")
logger.debug(base_nodes[3].get_content())

# Setup Query Engine


vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine()

base_vector_index = VectorStoreIndex(base_nodes)
base_query_engine = base_vector_index.as_query_engine()

# Run some Queries

# Example 1
query = "Tell me about yourself."

response = query_engine.query(query)
logger.newline()
logger.info("Semantic Query Response 1:")
display_jet_source_nodes(query, response)

base_response = base_query_engine.query(query)
logger.newline()
logger.info("Plain Query Response 1:")
display_jet_source_nodes(query, base_response)

# Example 2
query = "List your primary skills and achievements."

response = query_engine.query(query)
logger.newline()
logger.info("Semantic Query Response 2:")
display_jet_source_nodes(query, response)

base_response = base_query_engine.query(query)
logger.newline()
logger.info("Plain Query Response 2:")
display_jet_source_nodes(query, base_response)


logger.info("\n\n[DONE]", bright=True)
