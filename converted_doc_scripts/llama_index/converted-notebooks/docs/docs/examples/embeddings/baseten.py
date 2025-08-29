from jet.logger import CustomLogger
from llama_index.embeddings.baseten import BasetenEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/openai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Baseten Embeddings

This guide how to use performant open source embeddings and rerankers in the Baseten [library](https://www.baseten.co/library/tag/embedding/). First we need to install LlamaIndex and the Baseten dependencies.
"""
logger.info("# Baseten Embeddings")

# %pip install llama-index llama-index-embeddings-baseten

"""
Start a dedicated endpoint of your choice with the embedding you want [here](https://www.baseten.co/library/tag/embedding/) and paste your Baseten API key below.
"""
logger.info("Start a dedicated endpoint of your choice with the embedding you want [here](https://www.baseten.co/library/tag/embedding/) and paste your Baseten API key below.")


embed_model = BasetenEmbedding(
    model_id="BASETEN_MODEL_ID",  # 8 character string
    api_key="BASETEN_API_KEY",
)

embedding = embed_model.get_text_embedding("Hello, world!")

embeddings = embed_model.get_text_embedding_batch(
    ["Hello, world!", "Goodbye, world!"]
)
logger.debug(embeddings)

logger.info("\n\n[DONE]", bright=True)