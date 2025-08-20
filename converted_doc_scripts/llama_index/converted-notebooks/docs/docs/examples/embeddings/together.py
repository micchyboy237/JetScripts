from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.together import TogetherEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/together.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Together AI Embeddings

This notebook shows how to use `Together AI` for embeddings. Together AI provides access to many state-of-the-art embedding models.

Visit https://together.ai and sign up to get an API key.

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Together AI Embeddings")

# %pip install llama-index-embeddings-together

# !pip install llama-index


embed_model = TogetherEmbedding(
    model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key="..."
)

"""
## Get Embeddings
"""
logger.info("## Get Embeddings")

embeddings = embed_model.get_text_embedding("hello world")

logger.debug(len(embeddings))

logger.debug(embeddings[:5])

logger.info("\n\n[DONE]", bright=True)