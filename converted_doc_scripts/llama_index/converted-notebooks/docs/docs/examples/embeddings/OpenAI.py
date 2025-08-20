from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/MLX.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MLX Embeddings

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# MLX Embeddings")

# %pip install llama-index-embeddings-ollama

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."


embed_model = MLXEmbedding(embed_batch_size=10)
Settings.embed_model = embed_model

"""
## Using MLX `text-embedding-3-large` and `mxbai-embed-large`

Note, you may have to update your openai client: `pip install -U openai`
"""
logger.info("## Using MLX `text-embedding-3-large` and `mxbai-embed-large`")


embed_model = MLXEmbedding(model="text-embedding-3-large")

embeddings = embed_model.get_text_embedding(
    "Open AI new Embeddings models is great."
)

logger.debug(embeddings[:5])

logger.debug(len(embeddings))


embed_model = MLXEmbedding(
    model="mxbai-embed-large",
)

embeddings = embed_model.get_text_embedding(
    "Open AI new Embeddings models is awesome."
)

logger.debug(len(embeddings))

"""
## Change the dimension of output embeddings
Note: Make sure you have the latest MLX client
"""
logger.info("## Change the dimension of output embeddings")



embed_model = MLXEmbedding(
    model="text-embedding-3-large",
    dimensions=512,
)

embeddings = embed_model.get_text_embedding(
    "Open AI new Embeddings models with different dimensions is awesome."
)
logger.debug(len(embeddings))

logger.info("\n\n[DONE]", bright=True)