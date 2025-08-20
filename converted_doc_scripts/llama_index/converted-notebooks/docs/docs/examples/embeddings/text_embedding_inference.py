import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.text_embeddings_inference import (
TextEmbeddingsInference,
)
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/text_embedding_inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Text Embedding Inference

This notebook demonstrates how to configure `TextEmbeddingInference` embeddings.

The first step is to deploy the embeddings server. For detailed instructions, see the [official repository for Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference). Or [tei-gaudi repository](https://github.com/huggingface/tei-gaudi) if you are deploying on Habana Gaudi/Gaudi 2. 

Once deployed, the code below will connect to and submit embeddings for inference.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Text Embedding Inference")

# %pip install llama-index-embeddings-text-embeddings-inference

# !pip install llama-index



embed_model = TextEmbeddingsInference(
    model_name="BAAI/bge-large-en-v1.5",  # required for formatting inference text,
    timeout=60,  # timeout in seconds
    embed_batch_size=10,  # batch size for embedding
)

embeddings = embed_model.get_text_embedding("Hello World!")
logger.debug(len(embeddings))
logger.debug(embeddings[:5])

async def run_async_code_270de6b2():
    async def run_async_code_9760c351():
        embeddings = await embed_model.aget_text_embedding("Hello World!")
        return embeddings
    embeddings = asyncio.run(run_async_code_9760c351())
    logger.success(format_json(embeddings))
    return embeddings
embeddings = asyncio.run(run_async_code_270de6b2())
logger.success(format_json(embeddings))
logger.debug(len(embeddings))
logger.debug(embeddings[:5])

logger.info("\n\n[DONE]", bright=True)