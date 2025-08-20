import asyncio
from jet.transformers.formatters import format_json
from dotenv import load_dotenv, find_dotenv
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import asyncio
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/deepinfra.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# DeepInfra

With this integration, you can use the DeepInfra embeddings model to get embeddings for your text data. Here is the link to the [embeddings models](https://deepinfra.com/models/embeddings).

First, you need to sign up on the [DeepInfra website](https://deepinfra.com/) and get the API token. You can copy `model_ids` from the model cards and start using them in your code.

### Installation
"""
logger.info("# DeepInfra")

# !pip install llama-index llama-index-embeddings-deepinfra

"""
### Initialization
"""
logger.info("### Initialization")


_ = load_dotenv(find_dotenv())

model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-large-en-v1.5",  # Use custom model ID
    api_token="YOUR_API_TOKEN",  # Optionally provide token here
    normalize=True,  # Optional normalization
    text_prefix="text: ",  # Optional text prefix
    query_prefix="query: ",  # Optional query prefix
)

"""
### Synchronous Requests

#### Get Text Embedding
"""
logger.info("### Synchronous Requests")

response = model.get_text_embedding("hello world")
logger.debug(response)

"""
#### Batch Requests
"""
logger.info("#### Batch Requests")

texts = ["hello world", "goodbye world"]
response_batch = model.get_text_embedding_batch(texts)
logger.debug(response_batch)

"""
#### Query Requests
"""
logger.info("#### Query Requests")

query_response = model.get_query_embedding("hello world")
logger.debug(query_response)

"""
### Asynchronous Requests

#### Get Text Embedding
"""
logger.info("### Asynchronous Requests")

async def main():
    text = "hello world"
    async def run_async_code_4e58dd47():
        async def run_async_code_dc4cec8f():
            async_response = await model.aget_text_embedding(text)
            return async_response
        async_response = asyncio.run(run_async_code_dc4cec8f())
        logger.success(format_json(async_response))
        return async_response
    async_response = asyncio.run(run_async_code_4e58dd47())
    logger.success(format_json(async_response))
    logger.debug(async_response)


if __name__ == "__main__":

    asyncio.run(main())

"""
---

For any questions or feedback, please contact us at feedback@deepinfra.com.
"""
logger.info("For any questions or feedback, please contact us at feedback@deepinfra.com.")

logger.info("\n\n[DONE]", bright=True)