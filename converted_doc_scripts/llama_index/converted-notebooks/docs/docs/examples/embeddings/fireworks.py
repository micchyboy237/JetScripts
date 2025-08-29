from jet.logger import CustomLogger
from llama_index.embeddings.fireworks import FireworksEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/OllamaFunctionCallingAdapter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Fireworks Embeddings

This guide shows you how to use Fireworks Embeddings through [Fireworks Endpoints](https://readme.fireworks.ai/).

First, let's install LlamaIndex and the Fireworks dependencies
"""
logger.info("# Fireworks Embeddings")

# %pip install llama-index-embeddings-fireworks

# !pip install llama-index

"""
We can then query embeddings on Fireworks
"""
logger.info("We can then query embeddings on Fireworks")


embed_model = FireworksEmbedding(api_key="YOUR API KEY", embed_batch_size=10)

embeddings = embed_model.get_text_embedding("How do I sail to the moon?")
logger.debug(len(embeddings), embeddings[:10])

logger.info("\n\n[DONE]", bright=True)