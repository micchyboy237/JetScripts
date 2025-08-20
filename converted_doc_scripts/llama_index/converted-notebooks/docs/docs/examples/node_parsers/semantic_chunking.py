from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import (
SentenceSplitter,
SemanticSplitterNodeParser,
)
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_parsers/semantic_chunking.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Semantic Chunker

"Semantic chunking" is a new concept proposed Greg Kamradt in his video tutorial on 5 levels of embedding chunking: https://youtu.be/8OJC21T2SL4?t=1933.

Instead of chunking text with a **fixed** chunk size, the semantic splitter adaptively picks the breakpoint in-between sentences using embedding similarity. This ensures that a "chunk" contains sentences that are semantically related to each other. 

We adapted it into a LlamaIndex module.

Check out our notebook below!

Caveats:

- The regex primarily works for English sentences
- You may have to tune the breakpoint percentile threshold.

## Setup Data
"""
logger.info("# Semantic Chunker")

# %pip install llama-index-embeddings-ollama

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'pg_essay.txt'


documents = SimpleDirectoryReader(input_files=["pg_essay.txt"]).load_data()

"""
## Define Semantic Splitter
"""
logger.info("## Define Semantic Splitter")



# os.environ["OPENAI_API_KEY"] = "sk-..."

embed_model = MLXEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

base_splitter = SentenceSplitter(chunk_size=512)

nodes = splitter.get_nodes_from_documents(documents)

"""
### Inspecting the Chunks

Let's take a look at chunks produced by the semantic splitter.

#### Chunk 1: IBM 1401
"""
logger.info("### Inspecting the Chunks")

logger.debug(nodes[1].get_content())

"""
#### Chunk 2: Personal Computer + College
"""
logger.info("#### Chunk 2: Personal Computer + College")

logger.debug(nodes[2].get_content())

"""
#### Chunk 3: Finishing up College + Grad School
"""
logger.info("#### Chunk 3: Finishing up College + Grad School")

logger.debug(nodes[3].get_content())

"""
### Compare against Baseline

In contrast let's compare against the baseline with a fixed chunk size.
"""
logger.info("### Compare against Baseline")

base_nodes = base_splitter.get_nodes_from_documents(documents)

logger.debug(base_nodes[2].get_content())

"""
## Setup Query Engine
"""
logger.info("## Setup Query Engine")


vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine()

base_vector_index = VectorStoreIndex(base_nodes)
base_query_engine = base_vector_index.as_query_engine()

"""
### Run some Queries
"""
logger.info("### Run some Queries")

response = query_engine.query(
    "Tell me about the author's programming journey through childhood to college"
)

logger.debug(str(response))

for n in response.source_nodes:
    display_source_node(n, source_length=20000)

base_response = base_query_engine.query(
    "Tell me about the author's programming journey through childhood to college"
)

logger.debug(str(base_response))

for n in base_response.source_nodes:
    display_source_node(n, source_length=20000)

response = query_engine.query("Tell me about the author's experience in YC")

logger.debug(str(response))

base_response = base_query_engine.query(
    "Tell me about the author's experience in YC"
)

logger.debug(str(base_response))

logger.info("\n\n[DONE]", bright=True)