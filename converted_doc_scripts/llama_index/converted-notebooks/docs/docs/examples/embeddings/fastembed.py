from jet.logger import CustomLogger
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/clarifai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Qdrant FastEmbed Embeddings

LlamaIndex supports [FastEmbed](https://qdrant.github.io/fastembed/) for embeddings generation.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Qdrant FastEmbed Embeddings")

# %pip install llama-index-embeddings-fastembed

# %pip install llama-index

"""
To use this provider, the `fastembed` package needs to be installed.
"""
logger.info("To use this provider, the `fastembed` package needs to be installed.")

# %pip install fastembed

"""
The list of supported models can be found [here](https://qdrant.github.io/fastembed/examples/Supported_Models/).
"""
logger.info("The list of supported models can be found [here](https://qdrant.github.io/fastembed/examples/Supported_Models/).")


embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

embeddings = embed_model.get_text_embedding("Some text to embed.")
logger.debug(len(embeddings))
logger.debug(embeddings[:5])

logger.info("\n\n[DONE]", bright=True)