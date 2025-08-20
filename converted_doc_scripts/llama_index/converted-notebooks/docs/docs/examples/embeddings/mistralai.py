from jet.logger import CustomLogger
from llama_index.embeddings.mistralai import MistralAIEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/mistralai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MistralAI Embeddings

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# MistralAI Embeddings")

# %pip install llama-index-embeddings-mistralai

# !pip install llama-index


api_key = "YOUR API KEY"
model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=api_key)

embeddings = embed_model.get_text_embedding("La Plateforme - The Platform")

logger.debug(f"Dimension of embeddings: {len(embeddings)}")

embeddings[:5]

logger.info("\n\n[DONE]", bright=True)