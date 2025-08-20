from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.upstage import UpstageEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/upstage.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Upstage Embeddings

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Upstage Embeddings")

# %pip install llama-index-embeddings-upstage==0.2.1

# !pip install llama-index


os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"


embed_model = UpstageEmbedding()
Settings.embed_model = embed_model

"""
## Using Upstage Embeddings

Note, you may have to update your openai client: `pip install -U openai`
"""
logger.info("## Using Upstage Embeddings")


embed_model = UpstageEmbedding()

embeddings = embed_model.get_text_embedding(
    "Upstage new Embeddings models is great."
)

logger.debug(embeddings[:5])

logger.debug(len(embeddings))

embeddings = embed_model.get_query_embedding(
    "What are some great Embeddings model?"
)

logger.debug(embeddings[:5])

logger.debug(len(embeddings))

embeddings = embed_model.get_text_embedding_batch(
    [
        "Upstage new Embeddings models is awesome.",
        "Upstage LLM is also awesome.",
    ]
)

logger.debug(len(embeddings))

logger.debug(embeddings[0][:5])

logger.info("\n\n[DONE]", bright=True)