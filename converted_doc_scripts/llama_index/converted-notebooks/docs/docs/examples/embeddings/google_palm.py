from jet.logger import CustomLogger
from llama_index.embeddings.google import GooglePaLMEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/google_palm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Google PaLM Embeddings

**NOTE:** This example is deprecated. Please use the `GoogleGenAIEmbedding` class instead, detailed [here](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/google_genai.ipynb).

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Google PaLM Embeddings")

# %pip install llama-index-embeddings-google

# !pip install llama-index


model_name = "models/embedding-gecko-001"
api_key = "YOUR API KEY"

embed_model = GooglePaLMEmbedding(model_name=model_name, api_key=api_key)

embeddings = embed_model.get_text_embedding("Google PaLM Embeddings.")

logger.debug(f"Dimension of embeddings: {len(embeddings)}")

embeddings[:5]

logger.info("\n\n[DONE]", bright=True)