from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.cloudflare_workersai import CloudflareEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/cloudflare_workersai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Cloudflare Workers AI Embeddings

## Setup

Install library via pip
"""
logger.info("# Cloudflare Workers AI Embeddings")

# %pip install llama-index-embeddings-cloudflare-workersai

"""
To acess Cloudflare Workers AI, both Cloudflare account ID and API token are required. To get your account ID and API token, please follow the instructions on [this document](https://developers.cloudflare.com/workers-ai/get-started/rest-api/).
"""
logger.info("To acess Cloudflare Workers AI, both Cloudflare account ID and API token are required. To get your account ID and API token, please follow the instructions on [this document](https://developers.cloudflare.com/workers-ai/get-started/rest-api/).")

# import getpass

# my_account_id = getpass.getpass("Enter your Cloudflare account ID:\n\n")
# my_api_token = getpass.getpass("Enter your Cloudflare API token:\n\n")

"""
## Text embeddings example
"""
logger.info("## Text embeddings example")


my_embed = CloudflareEmbedding(
    account_id=my_account_id,
    auth_token=my_api_token,
    model="@cf/baai/bge-small-en-v1.5",
)

embeddings = my_embed.get_text_embedding("Why sky is blue")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
#### Embed in batches

As for batch size, Cloudflare's limit is a maximum of 100, as seen on 2024-03-31.
"""
logger.info("#### Embed in batches")

embeddings = my_embed.get_text_embedding_batch(
    ["Why sky is blue", "Why roses are red"]
)
logger.debug(len(embeddings))
logger.debug(len(embeddings[0]))
logger.debug(embeddings[0][:5])
logger.debug(embeddings[1][:5])

logger.info("\n\n[DONE]", bright=True)