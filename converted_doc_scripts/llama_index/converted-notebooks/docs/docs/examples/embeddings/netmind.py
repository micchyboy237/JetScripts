from jet.logger import CustomLogger
from llama_index.embeddings.netmind import NetmindEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Netmind AI Embeddings

This notebook shows how to use `Netmind AI` for embeddings.

Visit https://www.netmind.ai/ and sign up to get an API key.

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Netmind AI Embeddings")

# %pip install llama-index-embeddings-netmind

# !pip install llama-index


embed_model = NetmindEmbedding(
    model_name="BAAI/bge-m3", api_key="your-api-key"
)

"""
## Get Embeddings
"""
logger.info("## Get Embeddings")

embeddings = embed_model.get_text_embedding("hello world")

logger.debug(len(embeddings))

logger.debug(embeddings[:5])

logger.info("\n\n[DONE]", bright=True)