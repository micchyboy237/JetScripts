from jet.logger import CustomLogger
from llama_index.embeddings.llm_rails import LLMRailsEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/llm_rails.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LLMRails Embeddings

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# LLMRails Embeddings")

# %pip install llama-index-embeddings-llm-rails

# !pip install llama-index



api_key = os.environ.get("API_KEY", "your-api-key")
model_id = os.environ.get("MODEL_ID", "your-model-id")


embed_model = LLMRailsEmbedding(model_id=model_id, api_key=api_key)

embeddings = embed_model.get_text_embedding(
    "It is raining cats and dogs here!"
)

logger.info("\n\n[DONE]", bright=True)