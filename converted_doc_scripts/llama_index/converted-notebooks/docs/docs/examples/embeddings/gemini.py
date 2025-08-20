import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.embeddings.gemini import GeminiEmbedding
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

# Google Gemini Embeddings

**NOTE:** This example is deprecated. Please use the `GoogleGenAIEmbedding` class instead, detailed [here](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/google_genai.ipynb).

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Google Gemini Embeddings")

# %pip install llama-index-embeddings-gemini

# !pip install llama-index 'google-generativeai>=0.3.0' matplotlib


GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


model_name = "models/embedding-001"

embed_model = GeminiEmbedding(
    model_name=model_name, api_key=GOOGLE_API_KEY, title="this is a document"
)

embeddings = embed_model.get_text_embedding("Google Gemini Embeddings.")

logger.debug(f"Dimension of embeddings: {len(embeddings)}")

embeddings[:5]

embeddings = embed_model.get_query_embedding("Google Gemini Embeddings.")
embeddings[:5]

embeddings = embed_model.get_text_embedding(
    ["Google Gemini Embeddings.", "Google is awesome."]
)

logger.debug(f"Dimension of embeddings: {len(embeddings)}")
logger.debug(embeddings[0][:5])
logger.debug(embeddings[1][:5])

async def run_async_code_909df45e():
    async def run_async_code_9a7c314f():
        embedding = await embed_model.aget_text_embedding("Google Gemini Embeddings.")
        return embedding
    embedding = asyncio.run(run_async_code_9a7c314f())
    logger.success(format_json(embedding))
    return embedding
embedding = asyncio.run(run_async_code_909df45e())
logger.success(format_json(embedding))
logger.debug(embedding[:5])

async def async_func_37():
    embeddings = await embed_model.aget_text_embedding_batch(
        [
            "Google Gemini Embeddings.",
            "Google is awesome.",
            "Llamaindex is awesome.",
        ]
    )
    return embeddings
embeddings = asyncio.run(async_func_37())
logger.success(format_json(embeddings))
logger.debug(embeddings[0][:5])
logger.debug(embeddings[1][:5])
logger.debug(embeddings[2][:5])

async def run_async_code_280d9e0f():
    async def run_async_code_74e7d4ef():
        embedding = await embed_model.aget_query_embedding("Google Gemini Embeddings.")
        return embedding
    embedding = asyncio.run(run_async_code_74e7d4ef())
    logger.success(format_json(embedding))
    return embedding
embedding = asyncio.run(run_async_code_280d9e0f())
logger.success(format_json(embedding))
logger.debug(embedding[:5])

logger.info("\n\n[DONE]", bright=True)