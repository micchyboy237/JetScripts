from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.composability import QASummaryQueryEngineBuilder
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/JointQASummary.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Joint QA Summary Query Engine

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Joint QA Summary Query Engine")

# %pip install llama-index-llms-ollama

# !pip install llama-index

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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


reader = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/")
documents = reader.load_data()


gpt4 = MLXLlamaIndexLLMAdapter(temperature=0, model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

chatgpt = MLXLlamaIndexLLMAdapter(temperature=0, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")


query_engine_builder = QASummaryQueryEngineBuilder(
    llm=gpt4,
)
query_engine = query_engine_builder.build_from_documents(documents)

response = query_engine.query(
    "Can you give me a summary of the author's life?",
)

response = query_engine.query(
    "What did the author do growing up?",
)

response = query_engine.query(
    "What did the author do during his time in art school?",
)

logger.info("\n\n[DONE]", bright=True)