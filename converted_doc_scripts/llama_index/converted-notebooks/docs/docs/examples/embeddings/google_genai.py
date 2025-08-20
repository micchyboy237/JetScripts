import asyncio
from jet.transformers.formatters import format_json
from google.genai.types import EmbedContentConfig
from jet.logger import CustomLogger
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/gemini.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Google GenAI Embeddings

Using Google's `google-genai` package, LlamaIndex provides a `GoogleGenAIEmbedding` class that allows you to embed text using Google's GenAI models from both the Gemini and Vertex AI APIs.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Google GenAI Embeddings")

# %pip install llama-index-embeddings-google-genai


os.environ["GOOGLE_API_KEY"] = "..."

"""
## Setup

`GoogleGenAIEmbedding` is a wrapper around the `google-genai` package, which means it supports both Gemini and Vertex AI APIs out of that box.

You can pass in the `api_key` directly, or pass in a `vertexai_config` to use the Vertex AI API.

Other options include `embed_batch_size`, `model_name`, and `embedding_config`.

The default model is `text-embedding-004`.
"""
logger.info("## Setup")


embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100,
)

"""
## Usage

### Sync
"""
logger.info("## Usage")

embeddings = embed_model.get_text_embedding("Google Gemini Embeddings.")
logger.debug(embeddings[:5])
logger.debug(f"Dimension of embeddings: {len(embeddings)}")

embeddings = embed_model.get_query_embedding("Query Google Gemini Embeddings.")
logger.debug(embeddings[:5])
logger.debug(f"Dimension of embeddings: {len(embeddings)}")

embeddings = embed_model.get_text_embedding_batch(
    [
        "Google Gemini Embeddings.",
        "Google is awesome.",
        "Llamaindex is awesome.",
    ]
)
logger.debug(f"Got {len(embeddings)} embeddings")
logger.debug(f"Dimension of embeddings: {len(embeddings[0])}")

"""
### Async
"""
logger.info("### Async")

async def run_async_code_e6e00ebf():
    async def run_async_code_268e8da0():
        embeddings = await embed_model.aget_text_embedding("Google Gemini Embeddings.")
        return embeddings
    embeddings = asyncio.run(run_async_code_268e8da0())
    logger.success(format_json(embeddings))
    return embeddings
embeddings = asyncio.run(run_async_code_e6e00ebf())
logger.success(format_json(embeddings))
logger.debug(embeddings[:5])
logger.debug(f"Dimension of embeddings: {len(embeddings)}")

async def async_func_4():
    embeddings = await embed_model.aget_query_embedding(
        "Query Google Gemini Embeddings."
    )
    return embeddings
embeddings = asyncio.run(async_func_4())
logger.success(format_json(embeddings))
logger.debug(embeddings[:5])
logger.debug(f"Dimension of embeddings: {len(embeddings)}")

async def async_func_10():
    embeddings = await embed_model.aget_text_embedding_batch(
        [
            "Google Gemini Embeddings.",
            "Google is awesome.",
            "Llamaindex is awesome.",
        ]
    )
    return embeddings
embeddings = asyncio.run(async_func_10())
logger.success(format_json(embeddings))
logger.debug(f"Got {len(embeddings)} embeddings")
logger.debug(f"Dimension of embeddings: {len(embeddings[0])}")

logger.info("\n\n[DONE]", bright=True)