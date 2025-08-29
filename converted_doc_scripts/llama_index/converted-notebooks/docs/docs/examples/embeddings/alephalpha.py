from jet.logger import CustomLogger
from llama_index.embeddings.alephalpha import AlephAlphaEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/alephalpha.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Aleph Alpha Embeddings

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Aleph Alpha Embeddings")

# %pip install llama-index-embeddings-alephalpha

# !pip install llama-index


os.environ["AA_TOKEN"] = "your_token_here"

"""
#### With `luminous-base` embeddings.

- representation="Document": Use this for texts (documents) you want to store in your vector database
- representation="Query": Use this for search queries to find the most relevant documents in your vector database
- representation="Symmetric": Use this for clustering, classification, anomaly detection or visualisation tasks.
"""
logger.info("#### With `luminous-base` embeddings.")



embed_model = AlephAlphaEmbedding(
    model="luminous-base",
    representation="Query",
)

embeddings = embed_model.get_text_embedding("Hello Aleph Alpha!")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

embed_model = AlephAlphaEmbedding(
    model="luminous-base",
    representation="Document",
)

embeddings = embed_model.get_text_embedding("Hello Aleph Alpha!")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

logger.info("\n\n[DONE]", bright=True)