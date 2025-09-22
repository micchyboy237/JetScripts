from jet.logger import logger
from llama_index.embeddings.ollama import OllamaEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/ollama_embedding.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Ollama Embeddings

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Ollama Embeddings")

# %pip install llama-index-embeddings-ollama


ollama_embedding = OllamaEmbedding(
    model_name="embeddinggemma",
    base_url="http://localhost:11434",
)

"""
You can generate embeddings using one of several methods:

- `get_text_embedding_batch`
- `get_text_embedding`
- `get_query_embedding`

As well as async versions:
- `aget_text_embedding_batch`
- `aget_text_embedding`
- `aget_query_embedding`
"""
logger.info("You can generate embeddings using one of several methods:")

embeddings = ollama_embedding.get_text_embedding_batch(
    ["This is a passage!", "This is another passage"], show_progress=True
)
logger.debug(f"Got vectors of length {len(embeddings[0])}")
logger.debug(embeddings[0][:10])

embedding = ollama_embedding.get_text_embedding(
    "This is a piece of text!",
)
logger.debug(f"Got vectors of length {len(embedding)}")
logger.debug(embedding[:10])

embedding = ollama_embedding.get_query_embedding(
    "This is a query!",
)
logger.debug(f"Got vectors of length {len(embedding)}")
logger.debug(embedding[:10])

logger.info("\n\n[DONE]", bright=True)