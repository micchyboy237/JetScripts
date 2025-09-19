from jet.logger import CustomLogger
from llama_index.embeddings.anyscale import AnyscaleEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/OllamaFunctionCalling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Anyscale Embeddings

This guide shows you how to use Anyscale Embeddings through [Anyscale Endpoints](https://docs.endpoints.anyscale.com/).

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Anyscale Embeddings")

# %pip install llama-index-embeddings-anyscale

# !pip install llama-index


embed_model = AnyscaleEmbedding(
    api_key=ANYSCALE_ENDPOINT_TOKEN, embed_batch_size=10
)

embeddings = embed_model.get_text_embedding(
    "It is raining cats and dogs here!"
)
logger.debug(len(embeddings), embeddings[:10])

logger.info("\n\n[DONE]", bright=True)