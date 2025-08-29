from jet.transformers.formatters import format_json
from PIL import Image
from io import BytesIO
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.nvidia import NVIDIAMultiModal
import base64
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/nvidia_multi_modal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Modal LLM using NVIDIA endpoints for image reasoning

In this notebook, we show how to use NVIDIA MultiModal LLM class/abstraction for image understanding/reasoning.

We also show several functions we are now supporting for NVIDIA LLM:
* `complete` (both sync and async): for a single prompt and list of images
* `stream complete` (both sync and async): for steaming output of complete
"""
logger.info("# Multi-Modal LLM using NVIDIA endpoints for image reasoning")

# %pip install --upgrade --quiet llama-index-multi-modal-llms-nvidia llama-index-embeddings-nvidia llama-index-readers-file

# import getpass

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    logger.debug("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
#     nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith(
        "nvapi-"
    ), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

# import nest_asyncio

# nest_asyncio.apply()



llm = NVIDIAMultiModal()

"""
## Initialize `NVIDIAMultiModal` and Load Images from URLs
"""
logger.info("## Initialize `NVIDIAMultiModal` and Load Images from URLs")

image_urls = [
    "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",
    "https://www.visualcapitalist.com/wp-content/uploads/2023/10/US_Mortgage_Rate_Surge-Sept-11-1.jpg",
    "https://www.sportsnet.ca/wp-content/uploads/2023/11/CP1688996471-1040x572.jpg",
]

img_response = requests.get(image_urls[0])
img = Image.open(BytesIO(img_response.content))

image_url_documents = load_image_urls(image_urls)

"""
### Complete a prompt with a bunch of images
"""
logger.info("### Complete a prompt with a bunch of images")

response = llm.complete(
    prompt=f"What is this image?",
    image_documents=image_url_documents,
)

logger.debug(response)

llm.complete(
    prompt="tell me about this image",
    image_documents=image_url_documents,
)

"""
### Steam Complete a prompt with a bunch of images
"""
logger.info("### Steam Complete a prompt with a bunch of images")

stream_complete_response = llm.stream_complete(
    prompt=f"What is this image?",
    image_documents=image_url_documents,
)

for r in stream_complete_response:
    logger.debug(r.text, end="")

stream_complete_response = llm.stream_complete(
        prompt=f"What is this image?",
        image_documents=image_url_documents,
    )
logger.success(format_json(stream_complete_response))

last_element = None
async for last_element in stream_complete_response:
    pass

logger.debug(last_element)

"""
##  Passing an image as a base64 encoded string
"""
logger.info("##  Passing an image as a base64 encoded string")

imgr_content = base64.b64encode(
    requests.get(
        "https://helloartsy.com/wp-content/uploads/kids/cats/how-to-draw-a-small-cat/how-to-draw-a-small-cat-step-6.jpg"
    ).content
).decode("utf-8")

llm.complete(
    prompt="List models in image",
    image_documents=[ImageDocument(image=imgr_content, mimetype="jpeg")],
)

"""
##  Passing an image as an NVCF asset
If your image is sufficiently large or you will pass it multiple times in a chat conversation, you may upload it once and reference it in your chat conversation

See https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/assets.html for details about how upload the image.
"""
logger.info("##  Passing an image as an NVCF asset")


content_type = "image/jpg"
description = "example-image-from-lc-nv-ai-e-notebook"

create_response = requests.post(
    "https://api.nvcf.nvidia.com/v2/nvcf/assets",
    headers={
        "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
        "accept": "application/json",
        "Content-Type": "application/json",
    },
    json={"contentType": content_type, "description": description},
)
create_response.raise_for_status()

upload_response = requests.put(
    create_response.json()["uploadUrl"],
    headers={
        "Content-Type": content_type,
        "x-amz-meta-nvcf-asset-description": description,
    },
    data=img_response.content,
)
upload_response.raise_for_status()

asset_id = create_response.json()["assetId"]
asset_id

response = llm.stream_complete(
    prompt=f"Describe the image",
    image_documents=[
        ImageDocument(metadata={"asset_id": asset_id}, mimetype="png")
    ],
)

for r in response:
    logger.debug(r.text, end="")

"""
##  Passing images from local files
"""
logger.info("##  Passing images from local files")


image_documents = SimpleDirectoryReader("./tests/data/").load_data()

llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

"""
### Chat with of images
"""
logger.info("### Chat with of images")


llm.chat(
    [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": image_urls[1]},
            ],
        )
    ]
)


llm.chat(
    [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": image_urls[1]},
            ],
        )
    ]
)

llm.chat(
    [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe the image"},
                {
                    "type": "image_url",
                    "image_url": f'<img src="data:{content_type};asset_id,{asset_id}" />',
                },
            ],
        )
    ]
)

llm.chat(
    [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe the image"},
                {
                    "type": "image_url",
                    "image_url": f'<img src="data:{content_type};asset_id,{asset_id}" />',
                },
            ],
        )
    ]
)

"""
### Stream Chat a prompt with images
"""
logger.info("### Stream Chat a prompt with images")


streaming_resp = llm.stream_chat(
    [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": image_urls[1]},
            ],
        )
    ]
)

for r in streaming_resp:
    logger.debug(r.delta, end="")


resp = llm.stream_chat(
        [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this image:"},
                    {"type": "image_url", "image_url": image_urls[0]},
                ],
            )
        ]
    )
logger.success(format_json(resp))

last_element = None
async for last_element in resp:
    pass

logger.debug(last_element)

response = llm.stream_chat(
    [
        ChatMessage(
            role="user",
            content=f"""<img src="data:image/jpg;
            ,{asset_id}"/>""",
        )
    ]
)

for r in response:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)