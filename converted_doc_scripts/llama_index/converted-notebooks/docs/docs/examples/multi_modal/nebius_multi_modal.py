import asyncio
from jet.transformers.formatters import format_json
from PIL import Image
from io import BytesIO
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.nebius import NebiusMultiModal
from llama_index.multi_modal_llms.openai.utils import (
generate_openai_multi_modal_chat_message,
)
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

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/nebius_multi_modal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multimodal models with Nebius

This notebook demonstrates how to use multimodal models from [Nebius AI Studio](https://studio.nebius.ai/) with LlamaIndex. Nebius AI Studio implements all state-of-the-art multimodal models available for commercial use.

First, let's install LlamaIndex and dependencies of Nebius AI Studio. Since AI Studio uses MLX-compatible MLX, installation of the MLX Multimodal package inside Llama-index is also required.
"""
logger.info("# Multimodal models with Nebius")

# %pip install llama-index-multi-modal-llms-nebius llama-index matplotlib

"""
Upload your Nebius AI Studio key from system variables below or simply insert it. You can get it by registering for free at [Nebius AI Studio](https://auth.eu.nebius.com/ui/login) and issuing the key at [API Keys section](https://studio.nebius.ai/settings/api-keys)."
"""
logger.info("Upload your Nebius AI Studio key from system variables below or simply insert it. You can get it by registering for free at [Nebius AI Studio](https://auth.eu.nebius.com/ui/login) and issuing the key at [API Keys section](https://studio.nebius.ai/settings/api-keys)."")


NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")  # NEBIUS_API_KEY = ""

"""
##  Use Qwen to understand Images from URLs

## Initialize `NebiusMultiModal` and Load Images from URLs
"""
logger.info("##  Use Qwen to understand Images from URLs")




image_urls = [
    "https://townsquare.media/site/442/files/2018/06/wall-e-eve.jpg",
]

image_documents = load_image_urls(image_urls)

mm_llm = NebiusMultiModal(
    model="Qwen/Qwen2-VL-72B-Instruct",
    api_key=NEBIUS_API_KEY,
    max_new_tokens=300,
)


img_response = requests.get(image_urls[0])
logger.debug(image_urls[0])
img = Image.open(BytesIO(img_response.content))
plt.imshow(img)

"""
### Complete a prompt with a bunch of images
"""
logger.info("### Complete a prompt with a bunch of images")

complete_response = mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

logger.debug(complete_response)

"""
### Stream Complete a prompt with a bunch of images
"""
logger.info("### Stream Complete a prompt with a bunch of images")

stream_complete_response = mm_llm.stream_complete(
    prompt="give me more context for this image",
    image_documents=image_documents,
)

for r in stream_complete_response:
    logger.debug(r.delta, end="")

"""
### Chat through a list of chat messages
"""
logger.info("### Chat through a list of chat messages")


chat_msg_1 = generate_openai_multi_modal_chat_message(
    prompt="Describe the image as an alternative text",
    role="user",
    image_documents=image_documents,
)

chat_msg_2 = generate_openai_multi_modal_chat_message(
    prompt='The image features two animated characters from the movie "WALL-E."',
    role="assistant",
)

chat_msg_3 = generate_openai_multi_modal_chat_message(
    prompt="can I know more?",
    role="user",
)

chat_messages = [chat_msg_1, chat_msg_2, chat_msg_3]
chat_response = mm_llm.chat(
    messages=chat_messages,
)

for msg in chat_messages:
    logger.debug(msg.role, msg.content)

logger.debug(chat_response)

"""
### Stream Chat through a list of chat messages
"""
logger.info("### Stream Chat through a list of chat messages")

stream_chat_response = mm_llm.stream_chat(
    messages=chat_messages,
)

for r in stream_chat_response:
    logger.debug(r.delta, end="")

"""
### Async Complete
"""
logger.info("### Async Complete")

async def async_func_0():
    response_complete = mm_llm.complete(
        prompt="Describe the images as an alternative text",
        image_documents=image_documents,
    )
    return response_acomplete
response_acomplete = asyncio.run(async_func_0())
logger.success(format_json(response_acomplete))

logger.debug(response_acomplete)

"""
### Async Steam Complete
"""
logger.info("### Async Steam Complete")

async def async_func_0():
    response_stream_complete = mm_llm.stream_complete(
        prompt="Describe the images as an alternative text",
        image_documents=image_documents,
    )
    return response_astream_complete
response_astream_complete = asyncio.run(async_func_0())
logger.success(format_json(response_astream_complete))

for delta in response_stream_complete:
    logger.debug(delta.delta, end="")

"""
### Async Chat
"""
logger.info("### Async Chat")

async def async_func_0():
    chat_response = mm_llm.chat(
        messages=chat_messages,
    )
    return achat_response
achat_response = asyncio.run(async_func_0())
logger.success(format_json(achat_response))

logger.debug(achat_response)

"""
### Async stream Chat
"""
logger.info("### Async stream Chat")

async def async_func_0():
    stream_chat_response = mm_llm.stream_chat(
        messages=chat_messages,
    )
    return astream_chat_response
astream_chat_response = asyncio.run(async_func_0())
logger.success(format_json(astream_chat_response))

for delta in stream_chat_response:
    logger.debug(delta.delta, end="")

"""
##  Use Qwen to understand images from local files
"""
logger.info("##  Use Qwen to understand images from local files")


path_to_images = "/mnt/share/nebius/images"
image_documents = SimpleDirectoryReader(path_to_images).load_data()

mm_llm = NebiusMultiModal(
    model="Qwen/Qwen2-VL-72B-Instruct",
    api_key=NEBIUS_API_KEY,
    max_new_tokens=300,
)

response = mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)


for image_name in os.listdir(path_to_images):
    img = Image.open(os.path.join(path_to_images, image_name))
    plt.imshow(img)
    plt.show()

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)