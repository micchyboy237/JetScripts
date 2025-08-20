import asyncio
from jet.transformers.formatters import format_json
from IPython.display import Markdown, display
from PIL import Image
from io import BytesIO
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.mistralai import MistralAIMultiModal
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/mistral_multi_modal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Modal LLM using Mistral `pixtral-large` for image reasoning

In this notebook, we show how to use MistralAI MultiModal LLM class/abstraction for image understanding/reasoning.

We demonstrate following functions that are supported for MistralAI Pixtral Multimodal LLM:
* `complete` (both sync and async): for a single prompt and list of images
* `stream complete` (both sync and async): for steaming output of complete
"""
logger.info("# Multi-Modal LLM using Mistral `pixtral-large` for image reasoning")

# %pip install llama-index-multi-modal-llms-mistralai
# %pip install matplotlib


os.environ[
    "MISTRAL_API_KEY"
] = "<YOUR API KEY>"  # Your MistralAI API token here

"""
## Initialize `MistralAIMultiModal`

##
"""
logger.info("## Initialize `MistralAIMultiModal`")


mistralai_mm_llm = MistralAIMultiModal(
    model="pixtral-large-latest", max_new_tokens=1000
)

"""
## Load Images from URLs
"""
logger.info("## Load Images from URLs")



image_urls = [
    "https://tripfixers.com/wp-content/uploads/2019/11/eiffel-tower-with-snow.jpeg",
    "https://cdn.statcdn.com/Infographic/images/normal/30322.jpeg",
]

image_documents = load_image_urls(image_urls)

"""
### First Image
"""
logger.info("### First Image")


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
img_response = requests.get(image_urls[0], headers=headers)

logger.debug(image_urls[0])

img = Image.open(BytesIO(img_response.content))
plt.imshow(img)

"""
### Second Image
"""
logger.info("### Second Image")

img_response = requests.get(image_urls[1], headers=headers)

logger.debug(image_urls[1])

img = Image.open(BytesIO(img_response.content))
plt.imshow(img)

"""
### Complete a prompt with a bunch of images
"""
logger.info("### Complete a prompt with a bunch of images")

complete_response = mistralai_mm_llm.complete(
    prompt="Describe the images as an alternative text in a few words",
    image_documents=image_documents,
)

display(Markdown(f"{complete_response}"))

"""
### Steam Complete a prompt with a bunch of images
"""
logger.info("### Steam Complete a prompt with a bunch of images")

stream_complete_response = mistralai_mm_llm.stream_complete(
    prompt="give me more context for this images in a few words",
    image_documents=image_documents,
)

for r in stream_complete_response:
    logger.debug(r.delta, end="")

"""
### Async Complete
"""
logger.info("### Async Complete")

async def async_func_0():
    response_complete = mistralai_mm_llm.complete(
        prompt="Describe the images as an alternative text in a few words",
        image_documents=image_documents,
    )
    return response_acomplete
response_acomplete = asyncio.run(async_func_0())
logger.success(format_json(response_acomplete))

display(Markdown(f"{response_acomplete}"))

"""
### Async Steam Complete
"""
logger.info("### Async Steam Complete")

async def async_func_0():
    response_stream_complete = mistralai_mm_llm.stream_complete(
        prompt="Describe the images as an alternative text in a few words",
        image_documents=image_documents,
    )
    return response_astream_complete
response_astream_complete = asyncio.run(async_func_0())
logger.success(format_json(response_astream_complete))

for delta in response_stream_complete:
    logger.debug(delta.delta, end="")

"""
## Complete with Two images
"""
logger.info("## Complete with Two images")

image_urls = [
    "https://tripfixers.com/wp-content/uploads/2019/11/eiffel-tower-with-snow.jpeg",
    "https://assets.visitorscoverage.com/production/wp-content/uploads/2024/04/AdobeStock_626542468-min-1024x683.jpeg",
]

"""
### Lets Inspect the images.

### First Image
"""
logger.info("### Lets Inspect the images.")

img_response = requests.get(image_urls[0], headers=headers)

logger.debug(image_urls[0])

img = Image.open(BytesIO(img_response.content))
plt.imshow(img)

"""
### Second Image
"""
logger.info("### Second Image")

img_response = requests.get(image_urls[1], headers=headers)

logger.debug(image_urls[1])

img = Image.open(BytesIO(img_response.content))
plt.imshow(img)

image_documents_compare = load_image_urls(image_urls)

response_multi = mistralai_mm_llm.complete(
    prompt="What are the differences between two images?",
    image_documents=image_documents_compare,
)

display(Markdown(f"{response_multi}"))

"""
##  Load Images from local files
"""
logger.info("##  Load Images from local files")

# !wget 'https://www.boredpanda.com/blog/wp-content/uploads/2022/11/interesting-receipts-102-6364c8d181c6a__700.jpg' -O 'receipt.jpg'


img = Image.open("./receipt.jpg")
plt.imshow(img)


image_documents = SimpleDirectoryReader(
    input_files=["./receipt.jpg"]
).load_data()

response = mistralai_mm_llm.complete(
    prompt="Transcribe the text in the image",
    image_documents=image_documents,
)

display(Markdown(f"{response}"))

logger.info("\n\n[DONE]", bright=True)