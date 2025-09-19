from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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

# OllamaFunctionCalling Embeddings

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# OllamaFunctionCalling Embeddings")

# %pip install llama-index-embeddings-huggingface

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
Settings.embed_model = embed_model

"""
## Using OllamaFunctionCalling `text-embedding-3-large` and `mxbai-embed-large`

Note, you may have to update your openai client: `pip install -U openai`
"""
logger.info("## Using OllamaFunctionCalling `text-embedding-3-large` and `mxbai-embed-large`")


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

embeddings = embed_model.get_text_embedding(
    "Open AI new Embeddings models is great."
)

logger.debug(embeddings[:5])

logger.debug(len(embeddings))


embed_model = HuggingFaceEmbedding(
    model="mxbai-embed-large",
)

embeddings = embed_model.get_text_embedding(
    "Open AI new Embeddings models is awesome."
)

logger.debug(len(embeddings))

"""
## Change the dimension of output embeddings
Note: Make sure you have the latest OllamaFunctionCalling client
"""
logger.info("## Change the dimension of output embeddings")



embed_model = HuggingFaceEmbedding(
    model="text-embedding-3-large",
    dimensions=512,
)

embeddings = embed_model.get_text_embedding(
    "Open AI new Embeddings models with different dimensions is awesome."
)
logger.debug(len(embeddings))

logger.info("\n\n[DONE]", bright=True)