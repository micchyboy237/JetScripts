from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import FLAREInstructQueryEngine
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/flare_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# FLARE Query Engine

Adapted from the paper "Active Retrieval Augmented Generation"

Currently implements FLARE Instruct, which tells the LLM to generate retrieval instructions.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# FLARE Query Engine")

# %pip install llama-index-llms-ollama

# !pip install llama-index


Settings.llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0)
Settings.chunk_size = 512

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load Data
"""
logger.info("## Load Data")


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
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

logger.debug(response)

response = flare_query_engine.query(
    "Can you tell me about what the author did during his time at YC?"
)

logger.debug(response)

response = flare_query_engine.query(
    "Tell me about the author's life from childhood to adulthood"
)

logger.debug(response)

response = index_query_engine.query(
    "Can you tell me about the author's trajectory in the startup world?"
)

logger.debug(str(response))

response = index_query_engine.query(
    "Tell me about the author's life from childhood to adulthood"
)

logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)