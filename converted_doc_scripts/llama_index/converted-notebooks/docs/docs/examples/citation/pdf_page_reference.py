from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
SimpleDirectoryReader,
VectorStoreIndex,
download_loader,
RAKEKeywordTableIndex,
)
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
# Get References from PDFs 

This guide shows you how to use LlamaIndex to get in-line page number citations in the response (and the response is streamed).

This is a simple combination of using the page number metadata in our PDF loader along with our indexing/query abstractions to use this information.

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/citation/pdf_page_reference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Get References from PDFs")

# %pip install llama-index-llms-ollama

# !pip install llama-index



llm = MLX(temperature=0, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

"""
Load document and build index
"""
logger.info("Load document and build index")

reader = SimpleDirectoryReader(input_files=["./data/10k/lyft_2021.pdf"])
data = reader.load_data()

index = VectorStoreIndex.from_documents(data)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)

"""
Stream response with page citation
"""
logger.info("Stream response with page citation")

response = query_engine.query(
    "What was the impact of COVID? Show statements in bullet form and show"
    " page reference after each statement."
)
response.print_response_stream()

"""
Inspect source nodes
"""
logger.info("Inspect source nodes")

for node in response.source_nodes:
    logger.debug("-----")
    text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
    logger.debug(f"Text:\t {text_fmt} ...")
    logger.debug(f"Metadata:\t {node.node.metadata}")
    logger.debug(f"Score:\t {node.score:.3f}")

logger.info("\n\n[DONE]", bright=True)