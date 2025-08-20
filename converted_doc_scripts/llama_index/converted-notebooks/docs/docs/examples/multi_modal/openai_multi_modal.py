import asyncio
from jet.transformers.formatters import format_json
from PIL import Image
from io import BytesIO
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import (
ChatMessage,
ImageBlock,
TextBlock,
MessageRole,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
import matplotlib.pyplot as plt
import os
import requests
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/openai_multi_modal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Using MLX GPT-4V model for image reasoning

In this notebook, we show how to use the `MLX` LLM abstraction with GPT4V for image understanding/reasoning.

We also show several functions that are currently supported in the `MLX` LLM class when working with GPT4V:
* `complete` (both sync and async): for a single prompt and list of images
* `chat` (both sync and async): for multiple chat messages
* `stream complete` (both sync and async): for steaming output of complete
* `stream chat` (both sync and async): for steaming output of chat
"""
logger.info("# Using MLX GPT-4V model for image reasoning")

# %pip install llama-index-llms-ollama matplotlib

"""
##  Use GPT4V to understand Images from URLs
"""
logger.info("##  Use GPT4V to understand Images from URLs")


# OPENAI_API_KEY = "sk-..."  # Your MLX API token here
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

"""
## Initialize `MLXMultiModal` and Load Images from URLs

##
"""
logger.info("## Initialize `MLXMultiModal` and Load Images from URLs")


image_urls = [
    "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",
    "https://www.visualcapitalist.com/wp-content/uploads/2023/10/US_Mortgage_Rate_Surge-Sept-11-1.jpg",
    "https://i2-prod.mirror.co.uk/incoming/article7160664.ece/ALTERNATES/s1200d/FIFA-Ballon-dOr-Gala-2015.jpg",
]

openai_llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", max_new_tokens=300)


img_response = requests.get(image_urls[0])
logger.debug(image_urls[0])
img = Image.open(BytesIO(img_response.content))
plt.imshow(img)

"""
### Ask the model to describe what it sees
"""
logger.info("### Ask the model to describe what it sees")


msg = ChatMessage(
    role=MessageRole.USER,
    blocks=[
        TextBlock(text="Describe the images as an alternative text"),
        ImageBlock(url=image_urls[0]),
        ImageBlock(url=image_urls[1]),
    ],
)

response = openai_llm.chat(messages=[msg])

logger.debug(response)

"""
We can also stream the model response asynchronously
"""
logger.info("We can also stream the model response asynchronously")

async def run_async_code_7b06a55e():
    async def run_async_code_a3c93c66():
        async_resp = openai_llm.stream_chat(messages=[msg])
        return async_resp
    async_resp = asyncio.run(run_async_code_a3c93c66())
    logger.success(format_json(async_resp))
    return async_resp
async_resp = asyncio.run(run_async_code_7b06a55e())
logger.success(format_json(async_resp))
async for delta in async_resp:
    logger.debug(delta.delta, end="")

"""
##  Use GPT4V to understand images from local files
"""
logger.info("##  Use GPT4V to understand images from local files")

# %pip install llama-index-readers-file



img_path = Path().resolve() / "image.jpg"
response = requests.get(image_urls[-1])
with open(img_path, "wb") as file:
    file.write(response.content)

msg = ChatMessage(
    role=MessageRole.USER,
    blocks=[
        TextBlock(text="Describe the image as an alternative text"),
        ImageBlock(path=img_path, image_mimetype="image/jpeg"),
    ],
)

response = openai_llm.chat(messages=[msg])

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)